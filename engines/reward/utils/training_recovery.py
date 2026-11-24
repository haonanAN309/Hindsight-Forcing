# Modified for the Hindsight Forcing release.
# See the repository-root THIRD_PARTY_NOTICES.md for provenance and component licenses.
"""Same-world, per-rank recovery for Student/Critic S3 training.

This is a recovery snapshot, separate from full Student/EMA inference exports.
FSDP local shards are copied directly; no all-gather of model parameters occurs.
All ranks acknowledge their atomic files before rank 0 publishes the manifest.
The caller owns DataLoader replay and must restore RNG *after* replay.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist


VERSION = 1
COUNTERS = ("step", "student_updates", "critic_updates", "ema_updates")


def default_config_essentials(trainer):
    """Keep the resolved algorithm config; omit logging/recovery destinations."""
    config = trainer.config
    try:
        from omegaconf import OmegaConf
        values = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else (
            dict(config) if isinstance(config, dict) else vars(config))
    except ImportError:
        values = dict(config) if isinstance(config, dict) else vars(config)
    ignored = {"logdir", "wandb_save_dir", "wandb_key", "wandb_host", "wandb_entity",
               "wandb_project", "disable_wandb", "config_name", "tensorboard",
               "tensorboard_flush_secs", "runtime_progress", "log_cuda_memory",
               "resume_recovery", "recoverable_checkpoint_interval", "recovery_keep",
               "save_final_only", "log_iters", "no_visualize", "early_validation_steps"}
    return canonical({key: value for key, value in values.items()
                      if key not in ignored and not key.startswith("runtime_")})


def distributed():
    return dist.is_available() and dist.is_initialized()


def rank_world():
    return (dist.get_rank(), dist.get_world_size()) if distributed() else (0, 1)


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            checksum.update(block)
    return checksum.hexdigest()


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def _state(module):
    return {"parameters": {name: cpu_tree(value) for name, value in module.named_parameters()},
            "buffers": {name: cpu_tree(value) for name, value in module.named_buffers()}}


def _validate_state(module, saved, label):
    for collection, current in (("parameters", dict(module.named_parameters())),
                                ("buffers", dict(module.named_buffers()))):
        values = saved.get(collection, {})
        if values.keys() != current.keys():
            raise ValueError(f"{label} local {collection} names differ from recovery snapshot")
        for name, value in current.items():
            old = values[name]
            if not torch.is_tensor(old) or tuple(old.shape) != tuple(value.shape) or old.dtype != value.dtype:
                raise ValueError(f"{label} local tensor layout differs: {name}")


@torch.no_grad()
def _restore_state(module, saved):
    for collection, current in (("parameters", module.named_parameters()),
                                ("buffers", module.named_buffers())):
        for name, value in current:
            value.copy_(saved[collection][name].to(device=value.device))


def capture_rng(device=None):
    # Only the active rank device: get_rng_state_all() would initialize CUDA
    # contexts on other GPUs and waste memory in this multi-process setup.
    cpu_device = device is not None and not isinstance(device, int) and torch.device(device).type == "cpu"
    cuda = (torch.cuda.get_rng_state(device if device is not None else torch.cuda.current_device()).cpu()
            if not cpu_device and torch.cuda.is_available() else None)
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": cuda}


def restore_rng(state, device=None):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Recovery contains CUDA RNG, but CUDA is unavailable")
        torch.cuda.set_rng_state(state["torch_cuda"], device if device is not None else torch.cuda.current_device())


def _collect(status, phase):
    rank, world = rank_world()
    if distributed():
        gathered = [None] * world
        dist.all_gather_object(gathered, status)
    else:
        gathered = [status]
    errors = [dict(rank=i, error=value["error"]) for i, value in enumerate(gathered)
              if value.get("error")]
    if errors:
        raise RuntimeError(f"Recovery {phase} failed: {errors}")
    return gathered


def _barrier():
    if distributed():
        dist.barrier()


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_torch(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _committed_folders(root):
    results = []
    if not root.exists():
        return results
    for folder in root.glob("step_*"):
        if folder.is_symlink() or not folder.is_dir():
            continue
        manifest = folder / "manifest.json"
        if not manifest.is_file():
            continue
        data = json.loads(manifest.read_text())
        if (data.get("schema_version") == VERSION and data.get("state") == "complete"
                and folder.name == f'step_{int(data["step"]):06d}'):
            results.append((int(data["step"]), folder))
    return sorted(results, reverse=True)


def cleanup_recovery(trainer, retain=0):
    """Call collectively after successful final exports, or keep latest N saves."""
    if type(retain) is not int or retain < 0:
        raise ValueError("retain must be a nonnegative integer")
    rank, _ = rank_world()
    status = {}
    if rank == 0:
        try:
            for _, folder in _committed_folders(Path(trainer.output_path) / "_recovery")[retain:]:
                shutil.rmtree(folder)
        except Exception as exc:
            status["error"] = f"{type(exc).__name__}: {exc}"
    _collect(status, "cleanup")
    _barrier()


def save_recovery(trainer, dataloader_state, config_essentials=None, *, keep=1):
    """Collectively save one complete state after a fully finished global step."""
    if type(keep) is not int or keep < 1:
        raise ValueError("keep must be at least one")
    rank, world = rank_world()
    folder = Path(trainer.output_path) / "_recovery" / f"step_{trainer.step:06d}"
    manifest_path = folder / "manifest.json"
    status = {}
    try:
        if manifest_path.exists():
            raise RuntimeError(f"Refusing to overwrite committed recovery snapshot: {folder}")
        if type(dataloader_state.get("batches_consumed")) is not int or dataloader_state["batches_consumed"] < 0:
            raise ValueError("DataLoader recovery requires nonnegative batches_consumed")
        config = default_config_essentials(trainer) if config_essentials is None else canonical(config_essentials)
        loader = canonical(dataloader_state)
        if not config:
            raise ValueError("config_essentials must explicitly identify this training configuration")
        counters = {name: int(getattr(trainer, name, 0)) for name in COUNTERS}
        ema = trainer.generator_ema
        payload = {"schema_version": VERSION, "rank": rank, "world_size": world,
                   "config_essentials": config, "counters": counters,
                   "student": _state(trainer.model.generator), "critic": _state(trainer.model.fake_score),
                   "student_optimizer": cpu_tree(trainer.generator_optimizer.state_dict()),
                   "critic_optimizer": cpu_tree(trainer.critic_optimizer.state_dict()),
                   "ema": None if ema is None else {"decay": float(ema.decay), "shadow": cpu_tree(ema.state_dict())},
                   "dataloader_state": loader, "rng": capture_rng(getattr(trainer, "device", None))}
        folder.mkdir(parents=True, exist_ok=True)
        filename = f"rank_{rank:05d}.pt"
        rank_path = folder / filename
        _atomic_torch(rank_path, payload)
        status = {"rank": rank, "file": filename, "bytes": rank_path.stat().st_size,
                  "sha256": digest(rank_path), "counters": counters,
                  "config_essentials": config, "dataloader_state": loader}
        del payload
    except Exception as exc:
        status = {"rank": rank, "error": f"{type(exc).__name__}: {exc}"}
    ranks = _collect(status, "rank save")
    publication = {}
    if rank == 0:
        try:
            if any(item["config_essentials"] != ranks[0]["config_essentials"] or
                   item["counters"] != ranks[0]["counters"] or
                   item["dataloader_state"] != ranks[0]["dataloader_state"] for item in ranks):
                raise ValueError("Ranks disagree on recovery configuration/counters/DataLoader position")
            manifest = {"schema_version": VERSION, "state": "complete", "time": time.time(),
                        "step": trainer.step, "world_size": world,
                        "source_run": str(Path(trainer.output_path).resolve()),
                        "config_essentials": ranks[0]["config_essentials"],
                        "counters": ranks[0]["counters"], "ranks": ranks}
            _atomic_json(manifest_path, manifest)
        except Exception as exc:
            publication["error"] = f"{type(exc).__name__}: {exc}"
    _collect(publication, "manifest publication")
    _barrier()
    cleanup_recovery(trainer, retain=keep)
    return manifest_path


def _validate_optimizer(optimizer, saved, label):
    current = optimizer.state_dict()
    old_groups, groups = saved.get("param_groups", []), current["param_groups"]
    if len(old_groups) != len(groups):
        raise ValueError(f"{label} optimizer parameter groups differ")
    if any(len(a["params"]) != len(b["params"]) for a, b in zip(old_groups, groups)):
        raise ValueError(f"{label} optimizer local parameter count differs")


def load_recovery(trainer, checkpoint, config_essentials=None, *, ema_factory=None):
    """Collectively restore local shards and optimizer state, but not RNG yet.

    Returns dataloader_state and rng_state. Rebuild/replay the caller's iterator
    to batches_consumed, then call restore_rng(result['rng_state'], device).
    """
    rank, world = rank_world()
    checkpoint = Path(checkpoint)
    manifest_path = checkpoint / "manifest.json" if checkpoint.is_dir() else checkpoint
    folder = manifest_path.parent
    payload = manifest = None
    status = {}
    try:
        config_essentials = default_config_essentials(trainer) if config_essentials is None else canonical(config_essentials)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != VERSION or manifest.get("state") != "complete":
            raise ValueError("Not a committed recovery manifest")
        if manifest.get("world_size") != world:
            raise ValueError("Recovery requires the same world_size and rank layout")
        if manifest.get("config_essentials") != canonical(config_essentials):
            raise ValueError("Recovery training configuration differs")
        records = manifest.get("ranks", [])
        if len(records) != world or [r.get("rank") for r in records] != list(range(world)):
            raise ValueError("Recovery rank manifest is incomplete or misordered")
        for index, record in enumerate(records):
            if record.get("file") != f"rank_{index:05d}.pt":
                raise ValueError("Recovery rank filename is invalid")
            path = folder / record["file"]
            if not path.is_file() or path.stat().st_size != record["bytes"]:
                raise ValueError(f"Recovery rank file missing or truncated: {index}")
        entry = records[rank]
        rank_path = folder / entry["file"]
        if digest(rank_path) != entry["sha256"]:
            raise ValueError(f"Recovery rank checksum differs: {rank}")
        # This is a trusted locally produced training snapshot; Python/NumPy RNG
        # states intentionally contain ordinary pickle objects.
        payload = torch.load(rank_path, map_location="cpu", weights_only=False)
        if (payload.get("schema_version") != VERSION or payload.get("rank") != rank
                or payload.get("world_size") != world or
                payload.get("config_essentials") != canonical(config_essentials)
                or payload.get("counters") != manifest.get("counters")
                or payload.get("counters", {}).get("step") != manifest.get("step")):
            raise ValueError("Recovery rank payload disagrees with manifest")
        _validate_state(trainer.model.generator, payload["student"], "Student")
        _validate_state(trainer.model.fake_score, payload["critic"], "Critic")
        _validate_optimizer(trainer.generator_optimizer, payload["student_optimizer"], "Student")
        _validate_optimizer(trainer.critic_optimizer, payload["critic_optimizer"], "Critic")
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
    _collect(status, "load validation")
    status = {}
    try:
        _restore_state(trainer.model.generator, payload["student"])
        _restore_state(trainer.model.fake_score, payload["critic"])
        trainer.generator_optimizer.load_state_dict(payload["student_optimizer"])
        trainer.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        if payload["ema"] is None:
            trainer.generator_ema = None
        else:
            if trainer.generator_ema is None:
                if ema_factory is None:
                    from utils.distributed import EMA_FSDP
                    ema_factory = EMA_FSDP
                trainer.generator_ema = ema_factory(trainer.model.generator, decay=payload["ema"]["decay"])
            trainer.generator_ema.decay = payload["ema"]["decay"]
            trainer.generator_ema.load_state_dict(payload["ema"]["shadow"])
        for name, value in payload["counters"].items():
            setattr(trainer, name, value)
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
    _collect(status, "state restore")
    _barrier()
    result = {"manifest": str(manifest_path), "step": trainer.step,
              "source_run": manifest.get("source_run", str(folder.parent.parent)),
              "dataloader_state": payload["dataloader_state"], "rng_state": payload["rng"],
              "rng": payload["rng"]}
    del payload
    return result
