#!/usr/bin/env python3
"""One-GPU formal evaluation worker; no training modules/processes are changed.

Task JSON/JSONL: {id, prompt, seed, suite: short|motion|long}.  Models explicitly
select a native source tree, config, checkpoint key and inference adapter.
No architecture/weight fallback is allowed. Long generation uses bounded KV;
VAE decode streams through the native temporal cache into a single MP4.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import re
import socket
import sys
import time
import traceback

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
ADAPTERS = {"cf", "self_forcing", "hf", "raven", "reward_forcing", "sgf", "sgf_plus", "causvid"}
VIDEO = {"height": 480, "width": 832, "fps": 16}
LATENTS = {"short": 21, "motion": 21, "long": 240}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def file_identity(path):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def read_tasks(path):
    contents = Path(path).read_text()
    try:
        value = json.loads(contents)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in contents.splitlines() if line.strip()]
    if isinstance(value, dict):
        value = value.get("tasks", [value])
    if not isinstance(value, list) or not value:
        raise ValueError("Task file must contain a nonempty task list")
    return value


def validate_plan(manifest, model_id, tasks):
    models = manifest.get("models", [])
    if len({m["id"] for m in models}) != len(models):
        raise ValueError("Duplicate model identifiers")
    model = next(m for m in models if m["id"] == model_id)
    if not SAFE_ID.fullmatch(model_id) or model.get("adapter") not in ADAPTERS:
        raise ValueError("Unknown adapter or unsafe model id")
    for key in ("source_root", "config_path", "checkpoint_path"):
        if not Path(model.get(key, "")).is_absolute():
            raise ValueError(f"Model requires absolute {key}")
    if model.get("weight_key") not in ("generator", "generator_ema"):
        raise ValueError("Explicit generator/generator_ema checkpoint key required")
    if {**VIDEO, **manifest.get("video", {})} != VIDEO:
        raise ValueError("Formal protocol requires 832x480 at 16 FPS")
    policy_suites = inference_policy_suites(model)
    seen = set()
    roots = set()
    for task in tasks:
        if not SAFE_ID.fullmatch(task.get("id", "")) or task.get("suite") not in LATENTS:
            raise ValueError("Unsafe sample id or unsupported suite")
        identity = (task["suite"], task["id"])
        if identity in seen:
            raise ValueError("Duplicate task")
        seen.add(identity)
        if type(task.get("seed")) is not int or not 0 <= task["seed"] < 2**32:
            raise ValueError("seed must be an integer in [0,2**32)")
        if not isinstance(task.get("prompt"), str) or not task["prompt"].strip():
            raise ValueError("Nonempty prompt required")
        expected = LATENTS[task["suite"]]
        if task.get("latent_frames", expected) != expected:
            raise ValueError("Protocol latent length changed")
        roots.add(source_for_task(model, task))
        if task["suite"] in policy_suites:
            validate_long_policy(model)
    if len(roots) != 1:
        raise ValueError("One process must use one source root; separate short/long tasks")
    if not all(Path(root).is_absolute() for root in roots):
        raise ValueError("Source roots must be absolute")
    return model, Path(next(iter(roots)))


def source_for_task(model, task):
    return model.get("long_source_root", model["source_root"]) if task["suite"] == "long" else model["source_root"]


def inference_policy_suites(model):
    """Opt in to the existing rolling-KV backend for short cohorts as well.

    The legacy default remains long-only. This selects a cache/RoPE backend;
    sample length, denoising schedule, RNG draws and commit logic stay native.
    A short/long matched-policy experiment must explicitly share its source.
    """
    if "inference_policy_suites" not in model:
        return ("long",)
    suites = model["inference_policy_suites"]
    if (not isinstance(suites, list) or not suites
            or any(not isinstance(s, str) or s not in LATENTS for s in suites)
            or len(set(suites)) != len(suites) or "long" not in suites):
        raise ValueError("inference_policy_suites must be unique known suites including long")
    if model.get("adapter") in ("causvid", "reward_forcing"):
        raise ValueError("inference_policy_suites is only supported by rolling-KV adapters")
    if any(s in suites for s in ("short", "motion")):
        short_source = Path(model["source_root"]).resolve()
        long_source = Path(model.get("long_source_root", model["source_root"])).resolve()
        if short_source != long_source:
            raise ValueError("Short/long rolling-KV policy requires the same source_root and long_source_root")
    return tuple(suites)


def validate_long_policy(model):
    policy = model.get("long_policy", {})
    adapter = model["adapter"]
    if adapter == "causvid":
        if policy != {"kind": "native_overlap_extension", "overlap_latents": 3}:
            raise ValueError("CausVid must retain its native 3-latent overlap extension")
    elif adapter == "reward_forcing":
        if policy != {"kind": "ema_sink", "cache_latents": 9, "sink_latents": 3, "decay": 0.999}:
            raise ValueError("Reward Forcing requires native 9-frame EMA-Sink")
    else:
        if policy.get("kind") != "relative_rope_fifo":
            raise ValueError("Explicit bounded rolling-KV policy required")
        capacity, sink = policy.get("cache_latents"), policy.get("sink_latents")
        if type(capacity) is not int or type(sink) is not int or not 0 <= sink <= capacity - 3:
            raise ValueError("Invalid rolling KV geometry")
        if policy.get("position_mode") not in ("top_aligned", "contiguous"):
            raise ValueError("Explicit RoPE position mode required")
        if policy.get("train_latents") not in (12, 21) or capacity > 21:
            raise ValueError("Bounded KV exceeds the trained temporal range")
        if policy["position_mode"] == "top_aligned" and capacity > policy["train_latents"]:
            raise ValueError("Top-aligned cache exceeds its RoPE position ceiling")
        if adapter in ("sgf", "sgf_plus") and (capacity, sink) != (12, 3):
            raise ValueError("Native chunkwise SGF long inference uses 12 latent slots and 3 sink frames")
        if adapter == "sgf" and (policy["train_latents"], policy["position_mode"]) != (21, "top_aligned"):
            raise ValueError("Native SGF long script uses train_latents=21 and top_aligned RoPE")
        if adapter == "sgf_plus" and (policy["train_latents"], policy["position_mode"]) != (12, "contiguous"):
            raise ValueError("Native SGF+ long script uses train_latents=12 and contiguous RoPE")
    return policy


def selected_state(checkpoint, key):
    if not isinstance(checkpoint, Mapping) or key not in checkpoint:
        raise ValueError(f"Missing explicit checkpoint key {key}; no raw/EMA fallback")
    state = checkpoint[key]
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Empty/nonmapping checkpoint state")
    result = {}
    for name, tensor in state.items():
        if not isinstance(name, str):
            raise ValueError("Nonstring checkpoint parameter")
        normalized = ".".join(p for p in name.split(".") if p not in ("_fsdp_wrapped_module", "_orig_mod"))
        if normalized in result:
            raise ValueError("Checkpoint key collision after wrapper normalization")
        result[normalized] = tensor
    return result


def task_paths(root, model_id, task):
    relative = Path(task["suite"]) / model_id / task["id"]
    return {"video": Path(root) / "videos" / (str(relative) + ".mp4"),
            "metadata": Path(root) / "metadata" / (str(relative) + ".json"),
            "done": Path(root) / "done" / (str(relative) + ".json"),
            "state": Path(root) / "state" / (str(relative) + ".json")}


def complete(paths, fingerprint):
    try:
        record = json.loads(paths["metadata"].read_text())
        done = json.loads(paths["done"].read_text())
        return (record["state"] == "complete" and record["fingerprint"] == done["fingerprint"] == fingerprint
                and paths["video"].stat().st_size == done["video_bytes"] > 0)
    except (OSError, KeyError, ValueError):
        return False


def set_case_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def iter_decoded_pixels(vae, latent, piece_frames=3):
    """Native Wan decoder with persistent temporal state and bounded pixels.

    The released decode already visits latent frames one by one. Reproduce that
    loop without concatenating the full video on GPU. conv2 is a 1x1x1 projection;
    preserving its dtype and the decoder feature cache preserves the algorithm.
    Both CF and the older CausVid VAE expose these native decoder components.
    """
    import torch
    if latent.ndim != 5 or latent.shape[0] != 1:
        raise ValueError("Streaming VAE needs one [B,T,C,H,W] sample")
    network = vae.model
    if not all(hasattr(network, k) for k in ("clear_cache", "conv2", "decoder", "z_dim")):
        raise ValueError("Unsupported VAE; refusing independent-piece decode")
    if tuple(network.conv2.kernel_size) != (1, 1, 1):
        raise ValueError("Streaming requires the native temporal-pointwise conv2")
    scale = [vae.mean.to(device=latent.device, dtype=latent.dtype),
             1.0 / vae.std.to(device=latent.device, dtype=latent.dtype)]
    network.clear_cache()
    try:
        for start in range(0, latent.shape[1], piece_frames):
            z = latent[:, start:start + piece_frames].permute(0, 2, 1, 3, 4)
            z = z / scale[1].view(1, network.z_dim, 1, 1, 1) + scale[0].view(1, network.z_dim, 1, 1, 1)
            projected = network.conv2(z)
            decoded = []
            for index in range(projected.shape[2]):
                network._conv_idx = [0]
                decoded.append(network.decoder(projected[:, :, index:index + 1],
                    feat_cache=network._feat_map, feat_idx=network._conv_idx))
            pixels = torch.cat(decoded, 2).float().clamp_(-1, 1).permute(0, 2, 1, 3, 4)
            if not torch.isfinite(pixels).all().item():
                raise FloatingPointError("Nonfinite VAE output")
            yield pixels
    finally:
        network.clear_cache()


class VideoEncoder:
    def __init__(self, path, fps=16):
        import av
        self.av = av
        self.container = av.open(str(path), mode="w", options={"movflags": "+faststart"})
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width, self.stream.height = 832, 480
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "18", "preset": "medium"}
        self.stream.codec_context.thread_count = int(os.environ.get("BENCHMARK_CPU_THREADS", "2"))
        self.frames = 0

    def write(self, pixels):
        import torch
        if pixels.shape[0] != 1 or tuple(pixels.shape[2:]) != (3, 480, 832):
            raise ValueError(f"Invalid decoded geometry: {tuple(pixels.shape)}")
        arrays = ((pixels[0].permute(0, 2, 3, 1) * 0.5 + 0.5) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        for array in arrays:
            frame = self.av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in self.stream.encode(frame):
                self.container.mux(packet)
            self.frames += 1

    def close(self):
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()


@contextmanager
def stream_native_decode(vae, encoder, limit_frames=None, capture_frame=None):
    """Intercept only the final VAE call; native denoising/cache logic is untouched."""
    import torch
    original = vae.decode_to_pixel
    captured = {}
    def decode(latent, *args, **kwargs):
        count = 0
        for pixels in iter_decoded_pixels(vae, latent):
            end = count + pixels.shape[1]
            if capture_frame is not None and count <= capture_frame < end:
                captured["pixel"] = pixels[:, capture_frame-count:capture_frame-count+1].detach().cpu()
            if limit_frames is None or count < limit_frames:
                take = pixels.shape[1] if limit_frames is None else min(pixels.shape[1], limit_frames-count)
                encoder.write(pixels[:, :take])
            count = end
        captured["decoded_frames"] = count
        # Native pipelines only affine-transform this return value and hand it
        # back. Pixels have already been written; a scalar avoids retaining them.
        return torch.zeros((), device=latent.device, dtype=latent.dtype)
    vae.decode_to_pixel = decode
    try:
        yield captured
    finally:
        vae.decode_to_pixel = original


def build_pipeline(model, source_root, config, device):
    import torch
    if model["adapter"] == "causvid":
        from causvid.models.wan.causal_inference import InferencePipeline
        pipeline = InferencePipeline(config, device=device)
    else:
        from pipeline.causal_inference import CausalInferencePipeline
        pipeline = CausalInferencePipeline(config, device=device)
    steps = list(pipeline.denoising_step_list)
    if len(steps) != 4 or int(config.get("num_frame_per_block", 0)) != 3:
        raise ValueError("Formal evaluation requires four denoising forwards and 3-latent chunks")
    first_steps = getattr(pipeline, "denoising_step_list_first_chunk", None)
    if first_steps is not None and len(first_steps) != 4:
        raise ValueError("First chunk must also use exactly four denoising forwards")
    if bool(config.get("independent_first_frame", False)):
        raise ValueError("This evaluation protocol uses chunkwise, not framewise models")
    plus = bool(config.get("model_kwargs", {}).get("dual_full_expert", {}).get("enabled", False))
    if plus != (model["adapter"] == "sgf_plus"):
        raise ValueError("SGF+ memory-expert architecture must be enabled exactly for SGF+")
    state = torch.load(model["checkpoint_path"], map_location="cpu", weights_only=True, mmap=True)
    selected = selected_state(state, model["weight_key"])
    if model["adapter"] == "sgf_plus" and not any("memory" in k for k in selected):
        raise ValueError("SGF+ checkpoint has no memory-expert weights")
    pipeline.generator.load_state_dict(selected, strict=True)
    del selected, state
    pipeline.to(dtype=torch.bfloat16).eval()
    pipeline.generator.to(device)
    pipeline.text_encoder.to(device)
    pipeline.vae.to(device)
    if model["adapter"] == "reward_forcing":
        from pipeline.reward_forcing_training import reward_inference_pipeline
        pipeline.reward_sampler = reward_inference_pipeline(pipeline.generator, pipeline.scheduler, config)
    return pipeline


def causvid_extension(pipeline, noise, prompt, encoder, target_frames):
    """Released CausVid continuation: re-encode boundary pixel + last two latents."""
    import torch
    start_latents = None
    passes = 0
    while encoder.frames < target_frames:
        current_noise = noise if passes == 0 else torch.randn_like(noise)
        remaining = target_frames - encoder.frames
        take = min(72, remaining)  # native 81 frames minus 9-frame overlap
        with stream_native_decode(pipeline.vae, encoder, limit_frames=take, capture_frame=72) as captured:
            _, latents = pipeline.inference(current_noise, [prompt], start_latents=start_latents, return_latents=True)
        if not torch.isfinite(latents).all().item():
            raise FloatingPointError("Nonfinite CausVid continuation latents")
        if encoder.frames < target_frames:
            pixel = captured["pixel"].to(device=noise.device, dtype=noise.dtype).transpose(1, 2)
            vae = pipeline.vae
            scale = [vae.mean.to(pixel), 1.0 / vae.std.to(pixel)]
            first = vae.model.encode(pixel, scale).float().permute(0, 2, 1, 3, 4).to(noise.dtype)
            start_latents = torch.cat((first, latents[:, -2:]), 1)
        passes += 1
    return {"native_extension_passes": passes, "native_overlap_latents": 3}


def sample_video(pipeline, model, task, encoder):
    import torch
    device = next(pipeline.generator.parameters()).device
    latent_frames = LATENTS[task["suite"]]
    # Full initial noise allocation (48 MiB even for 60 s) preserves the native
    # seeded draw ordering; it is not a video-sized decoded pixel allocation.
    noise_frames = 21 if model["adapter"] == "causvid" else latent_frames
    noise = torch.randn([1, noise_frames, 16, 60, 104], device=device, dtype=torch.bfloat16)
    noise_sha = hashlib.sha256(noise.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    extra = {}
    if model["adapter"] == "causvid" and task["suite"] == "long":
        extra = causvid_extension(pipeline, noise, task["prompt"], encoder, 4 * latent_frames - 3)
    elif model["adapter"] == "reward_forcing":
        conditional = pipeline.text_encoder(text_prompts=[task["prompt"]])
        latent, _, _ = pipeline.reward_sampler.inference_with_trajectory(noise, **conditional)
        if not torch.isfinite(latent).all().item():
            raise FloatingPointError("Nonfinite Reward Forcing latents")
        for pixels in iter_decoded_pixels(pipeline.vae, latent):
            encoder.write(pixels)
        extra.update(pipeline.reward_sampler.last_runtime_stats)
    else:
        kwargs = {"noise": noise, "text_prompts": [task["prompt"]], "return_latents": True}
        if task["suite"] in inference_policy_suites(model):
            policy = validate_long_policy(model)
            if "long_video" not in inspect.signature(pipeline.inference).parameters:
                raise ValueError("Selected source lacks native rolling-KV support")
            kwargs.update(long_video=True, kv_cache_max_frames=policy["cache_latents"],
                          kv_cache_sink=policy["sink_latents"], kv_cache_train_frames=policy["train_latents"],
                          kv_cache_position_mode=policy["position_mode"])
            if "inference_policy_suites" in model:
                extra["effective_inference_policy"] = dict(policy)
        with stream_native_decode(pipeline.vae, encoder) as captured:
            _, latent = pipeline.inference(**kwargs)
        if tuple(latent.shape) != tuple(noise.shape) or not torch.isfinite(latent).all().item():
            raise FloatingPointError("Invalid/nonfinite native latents")
        extra["decoded_frames"] = captured["decoded_frames"]
    expected = 4 * latent_frames - 3
    if encoder.frames != expected:
        raise ValueError(f"Decoded frame count {encoder.frames} != {expected}")
    return {"initial_noise_sha256": noise_sha, "latent_frames": latent_frames,
            "frames": expected, "duration_seconds": expected / 16, **extra}


def run(args):
    manifest = json.loads(Path(args.manifest).read_text())
    tasks = read_tasks(args.task_file)
    model, source_root = validate_plan(manifest, args.model_id, tasks)
    if args.validate_only:
        print(json.dumps({"state": "validated", "model": model["id"], "source_root": str(source_root), "tasks": len(tasks)}))
        return
    output_root = Path(args.output_root or manifest["output_root"]).resolve()
    source_root = source_root.resolve()
    os.chdir(Path(manifest.get("asset_root", source_root)).resolve())
    sys.path.insert(0, str(source_root))
    import torch
    from omegaconf import OmegaConf
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU is required")
    torch.set_num_threads(int(os.environ.get("BENCHMARK_CPU_THREADS", "2")))
    torch.set_grad_enabled(False)
    torch.cuda.set_device(0)
    defaults = source_root / "configs/default_config.yaml"
    if not defaults.exists():
        defaults = source_root / "configs/default.yaml"
    config = OmegaConf.load(model["config_path"])
    if defaults.exists() and model["adapter"] != "causvid":
        config = OmegaConf.merge(OmegaConf.load(defaults), config)
    config = OmegaConf.merge(config, model.get("config_overrides", {}))
    config_plain = OmegaConf.to_container(config, resolve=True)
    checkpoint = file_identity(model["checkpoint_path"])
    paths_to_hash = ["pipeline/causal_inference.py", "pipeline/reward_forcing_training.py", "utils/wan_wrapper.py",
                     "wan/modules/causal_model.py", "wan/modules/vae.py", "causvid/models/wan/causal_inference.py",
                     "causvid/models/wan/wan_wrapper.py", "causvid/models/wan/causal_model.py"]
    hashes = {p: hashlib.sha256((source_root / p).read_bytes()).hexdigest() for p in paths_to_hash if (source_root / p).exists()}
    common = {"model": model, "checkpoint": checkpoint, "config": config_plain, "source_hashes": hashes,
              "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "video": VIDEO}
    pending = []
    for task in tasks:
        paths = task_paths(output_root, model["id"], task)
        fingerprint = digest({**common, "task": task})
        if not complete(paths, fingerprint):
            pending.append((task, paths, fingerprint))
    if not pending:
        return
    runtime = {"gpu": torch.cuda.get_device_name(),
               "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda}
    for task, paths, fingerprint in pending:
        atomic_json(paths["state"], {"state": "loading_model", "updated_at": utc_now(), "fingerprint": fingerprint, **runtime})
    load_start = time.perf_counter()
    set_case_seed(0)
    try:
        pipeline = build_pipeline(model, source_root, config, torch.device("cuda:0"))
    except Exception:
        error = traceback.format_exc()
        for task, paths, fingerprint in pending:
            atomic_json(paths["state"], {"state": "failed", "phase": "loading_model", "updated_at": utc_now(),
                "model_id": model["id"], "sample_id": task["id"], "suite": task["suite"],
                "fingerprint": fingerprint, "traceback": error, **runtime})
        raise
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    for task, paths, fingerprint in pending:
        start = time.perf_counter()
        state = {"model_id": model["id"], "sample_id": task["id"], "suite": task["suite"], "fingerprint": fingerprint, **runtime}
        calls = [0]
        expert_calls = {"memory": 0, "generation": 0, "unspecified": 0}
        def progress(module, positional, kwargs):
            calls[0] += 1
            role = "unspecified" if "use_memory_expert" not in kwargs else ("memory" if kwargs["use_memory_expert"] else "generation")
            expert_calls[role] += 1
            if calls[0] == 1 or calls[0] % 5 == 0:
                atomic_json(paths["state"], {**state, "state": "generating", "updated_at": utc_now(), "generator_forward_calls": calls[0]})
        handle = pipeline.generator.register_forward_pre_hook(progress, with_kwargs=True)
        paths["video"].parent.mkdir(parents=True, exist_ok=True)
        partial = paths["video"].with_name(paths["video"].stem + f".partial.{os.getpid()}.mp4")
        encoder = None
        try:
            set_case_seed(task["seed"])
            torch.cuda.reset_peak_memory_stats()
            encoder = VideoEncoder(partial)
            details = sample_video(pipeline, model, task, encoder)
            if model["adapter"] == "sgf_plus":
                chunks = LATENTS[task["suite"]] // 3
                if expert_calls != {"memory": chunks, "generation": chunks * 4, "unspecified": 0}:
                    raise RuntimeError(f"SGF+ expert routing violated: {expert_calls}")
            encoder.close()
            encoder = None
            torch.cuda.synchronize()
            os.replace(partial, paths["video"])
            record = {**common, **state, **details, "task": task, "sample": task, "state": "complete", "completed_at": utc_now(),
                      "video_path": str(paths["video"]), "video_bytes": paths["video"].stat().st_size,
                      "generator_forward_calls": calls[0], "load_seconds": load_seconds,
                      "expert_forward_calls": expert_calls,
                      "total_sample_seconds": time.perf_counter()-start,
                      "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                      "latent_saved": False, "decode": "native_temporal_cache_streamed", "encoding": "h264_crf18_medium"}
            atomic_json(paths["metadata"], record)
            atomic_json(paths["done"], {"fingerprint": fingerprint, "video_bytes": record["video_bytes"], "completed_at": record["completed_at"]})
            atomic_json(paths["state"], {**state, "state": "complete", "updated_at": utc_now()})
            print(json.dumps({"model_id": model["id"], "sample_id": task["id"], "video_path": str(paths["video"]), "metadata": str(paths["metadata"]), "state": "complete"}), flush=True)
        except Exception:
            atomic_json(paths["state"], {**state, "state": "failed", "updated_at": utc_now(), "traceback": traceback.format_exc()})
            raise
        finally:
            handle.remove()
            if encoder is not None:
                encoder.close()
            pipeline.vae.model.clear_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--validate-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
