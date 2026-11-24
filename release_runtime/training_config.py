"""CPU-only training configuration for the released native engines."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from omegaconf import OmegaConf

RELEASE_ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "hf": ("hf", "trainer.distillation", 5),
    "raven": ("raven", "trainer.baseline_distillation", 2),
    "reward_forcing": ("reward", "trainer.baseline_distillation", 5),
    "sgf": ("sgf", "trainer.matched_runtime", 5),
    "sgf_plus": ("sgf_plus", "trainer.matched_runtime", 5),
}


def positive_integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = int(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if str(number) != str(value) or number < 1:
        raise ValueError(f"{name} must be a positive integer")
    return number


def accumulation_for_batch(global_batch, world_size, micro_batch, requested=None):
    """Keep the optimizer-update batch fixed for the current deployment."""
    global_batch = positive_integer(global_batch, "global batch")
    world_size = positive_integer(world_size, "WORLD_SIZE")
    micro_batch = positive_integer(micro_batch, "micro batch")
    divisor = world_size * micro_batch
    if global_batch % divisor:
        raise ValueError(
            f"Global batch {global_batch} is not divisible by WORLD_SIZE "
            f"times micro batch ({divisor}); choose a compatible micro batch."
        )
    accumulation = global_batch // divisor
    if requested is not None and positive_integer(requested, "accumulation steps") != accumulation:
        raise ValueError(
            f"Requested accumulation does not preserve global batch {global_batch}; "
            f"this deployment requires {accumulation}."
        )
    return accumulation


def engine_files(method, release_root=RELEASE_ROOT):
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {', '.join(METHODS)}")
    engine_name, trainer_module, _ = METHODS[method]
    engine = Path(release_root) / "engines" / engine_name
    default = engine / "configs" / "default.yaml"
    if not default.is_file():
        default = engine / "configs" / "default_config.yaml"
    config_name = engine_name
    configuration = engine / "configs" / f"{config_name}.yaml"
    return engine, default, configuration, trainer_module


def build_training_config(method, *, micro_batch=None, accumulation_steps=None,
                          data_path=None, init_checkpoint=None, output_dir=None,
                          seed=None, reward_model_path=None, reward_base_model_path=None,
                          smoke_test=False,
                          env: Mapping[str, str] | None = None, release_root=RELEASE_ROOT):
    """Merge native settings and add runtime flags without loading model code.

    Relative paths are interpreted from the release root, which is also the
    working directory of the training launcher. No files or directories are
    created here; this function is safe to use for a dry run.
    """
    environment = os.environ if env is None else env
    engine, default, configuration, _ = engine_files(method, release_root)
    for path in (default, configuration):
        if not path.is_file():
            raise FileNotFoundError(f"Missing release configuration: {path}")
    config = OmegaConf.merge(OmegaConf.load(default), OmegaConf.load(configuration))
    world_size = positive_integer(environment.get("WORLD_SIZE", "1"), "WORLD_SIZE")
    batch = positive_integer(config.batch_size if micro_batch is None else micro_batch, "micro batch")
    if int(config.total_batch_size) != 64:
        raise ValueError("The released training protocol uses global batch 64")
    config.batch_size = batch
    config.gradient_accumulation_steps = accumulation_for_batch(
        config.total_batch_size, world_size, batch, accumulation_steps)
    config.image_or_video_shape[0] = batch
    expected_ratio = METHODS[method][2]
    if int(config.dfake_gen_update_ratio) != expected_ratio:
        raise ValueError("Native Student/Critic update ratio differs from the released protocol")
    if int(config.max_steps) != 150 * expected_ratio:
        raise ValueError("The released configurations must budget 150 Student updates")
    if method != "hf" and int(config.target_student_updates) != 150:
        raise ValueError("Baseline configuration must target 150 Student updates")
    if int(config.get("critic_warmup_steps", 0)) != 0:
        raise ValueError("The released protocol has no additional Critic warmup")
    if not config.save_student_and_ema:
        raise ValueError("The release saves both ordinary Student and EMA weights")
    if method == "hf":
        if not (config.ste_training and config.ste_history_grad and float(config.ste_alpha) == 1):
            raise ValueError("HF requires differentiable history with alpha 1")
        if not config.same_step_across_blocks or config.last_step_only:
            raise ValueError("HF shares a random selected step across chunks")
        supported_ste_options = {"ste_training", "ste_history_grad", "ste_alpha"}
        if any(key.startswith("ste_") and key not in supported_ste_options for key in config):
            raise ValueError("Unsupported option in the released HF training configuration")
    config.generator_ckpt = str(init_checkpoint or "checkpoints/initialization/causal_ode.pt")
    if data_path is not None:
        config.data_path = str(data_path)
    config.logdir = str(output_dir or Path("outputs") / method)
    if seed is not None:
        if isinstance(seed, bool) or not 0 <= int(seed) < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        config.seed = int(seed)
    if reward_model_path is not None or reward_base_model_path is not None:
        if method != "reward_forcing":
            raise ValueError("Reward model paths apply only to reward_forcing")
        if reward_model_path is not None:
            config.reward_model_path = str(reward_model_path)
        if reward_base_model_path is not None:
            config.reward_base_model_path = str(reward_base_model_path)
    # Native trainer command-line fields. Reporting remains local by default.
    config.no_save = False
    config.no_visualize = True
    config.tf = False
    config.config_name = method
    config.disable_wandb = True
    config.wandb_save_dir = ""
    config.wandb_host = "https://api.wandb.ai"
    config.wandb_key = ""
    config.wandb_entity = ""
    config.wandb_project = ""
    if type(smoke_test) is not bool:
        raise ValueError("smoke_test must be a boolean")
    if smoke_test:
        # Validate the paper configuration above before changing only the run
        # budget and EMA start. Model, rollout, loss and effective batch stay
        # identical to the corresponding full training entry.
        config.release_smoke_test = True
        config.max_steps = expected_ratio
        if method == "hf":
            config.ema_start_step = 0
        else:
            config.target_student_updates = 1
            config.ema_start_student_update = 0
        destination = Path(config.logdir).expanduser()
        if not destination.name or destination.name in (".", ".."):
            raise ValueError("Smoke-test output must have a directory name")
        if not destination.name.endswith("_smoke"):
            destination = destination.with_name(destination.name + "_smoke")
        config.logdir = str(destination)
    return config


def training_plan(method, config, *, env=None):
    environment = os.environ if env is None else env
    ratio = int(config.dfake_gen_update_ratio)
    steps = int(config.max_steps)
    student_updates = (int(config.target_student_updates)
                       if "target_student_updates" in config
                       else (steps + ratio - 1) // ratio)
    return {
        "method": method,
        "smoke_test": bool(config.get("release_smoke_test", False)),
        "student_updates": student_updates,
        "critic_updates": steps,
        "global_steps": steps,
        "global_batch": int(config.total_batch_size),
        "world_size": positive_integer(environment.get("WORLD_SIZE", "1"), "WORLD_SIZE"),
        "micro_batch": int(config.batch_size),
        "accumulation_steps": int(config.gradient_accumulation_steps),
        "config": OmegaConf.to_container(config, resolve=True),
    }


def validate_training_output(config, release_root=RELEASE_ROOT):
    """A short test must never reuse a full run or overwrite a previous test."""
    if not config.get("release_smoke_test", False):
        return
    destination = Path(config.logdir).expanduser()
    if not destination.is_absolute():
        destination = Path(release_root) / destination
    if not destination.name.endswith("_smoke"):
        raise ValueError("Smoke-test output directory must end with _smoke")
    if destination.is_symlink():
        raise ValueError("Smoke-test output must not be a symbolic link")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(
            f"Smoke-test output already exists and is not empty: {config.logdir}; "
            "choose a new --output-dir"
        )


def validate_distributed_environment(env=None):
    environment = os.environ if env is None else env
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    missing = [name for name in required if not environment.get(name)]
    if missing:
        raise ValueError("Launch training with torchrun; missing " + ", ".join(missing))
    world = positive_integer(environment["WORLD_SIZE"], "WORLD_SIZE")
    rank, local_rank = int(environment["RANK"]), int(environment["LOCAL_RANK"])
    port = positive_integer(environment["MASTER_PORT"], "MASTER_PORT")
    if not 0 <= rank < world or local_rank < 0 or port > 65535:
        raise ValueError("Invalid torchrun rank or rendezvous port")


def validate_training_assets(method, config, release_root=RELEASE_ROOT):
    root = Path(release_root)
    for label, name in (("training prompts", config.data_path),
                        ("CF S2 initialization", config.generator_ckpt)):
        path = Path(name)
        path = path if path.is_absolute() else root / path
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty {label}: {name}")
    if method == "reward_forcing":
        for name in (config.reward_model_path, config.reward_base_model_path):
            path = Path(name)
            path = path if path.is_absolute() else root / path
            if not path.is_dir():
                raise FileNotFoundError(f"Missing reward-model directory: {name}")
