#!/usr/bin/env python3
"""Launch a native HF/baseline trainer, or inspect its resolved CPU-only plan."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

from release_runtime.training_config import (
    METHODS, RELEASE_ROOT, build_training_config, engine_files, training_plan,
    validate_distributed_environment, validate_training_assets, validate_training_output,
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--method", choices=METHODS, default="hf")
    result.add_argument("--dry-run", action="store_true", help="Print settings without importing PyTorch, loading weights, or starting training")
    result.add_argument("--smoke-test", action="store_true", help="Run one real Student update and the method's matching Critic updates, save ordinary/EMA weights, and append _smoke to the output directory")
    result.add_argument("--data-path", help="Prompt text file; relative paths are rooted at this release")
    result.add_argument("--init-checkpoint", help="CF S2 checkpoint (default: checkpoints/initialization/causal_ode.pt)")
    result.add_argument("--output-dir", help="Output directory (default: outputs/<method>)")
    result.add_argument("--micro-batch", type=int, help="Samples per rank; accumulation is derived to retain global batch 64")
    result.add_argument("--accumulation-steps", type=int, help="Optional explicit accumulation; must preserve global batch 64")
    result.add_argument("--seed", type=int, help="Training seed; defaults to the released configuration")
    result.add_argument("--reward-model-path", help="VideoReward checkpoint directory for Reward Forcing")
    result.add_argument("--reward-base-model-path", help="Qwen2-VL base directory for Reward Forcing")
    return result


def initialize_smoke_ema(native, trainer, method, config):
    """HF normally starts EMA later; short tests start from the loaded Student."""
    if config.get("release_smoke_test", False) and method == "hf":
        # The inherited HF constructor creates its initial EMA before loading
        # the S2 checkpoint. Rebuild it after that load for the shortened run.
        trainer.generator_ema = None
        trainer.generator_ema = native.EMA_FSDP(
            trainer.model.generator, decay=config.ema_weight)


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    options = vars(args).copy()
    options.pop("dry_run")
    method = options.pop("method")
    try:
        config = build_training_config(method, **options)
        if args.dry_run:
            print(json.dumps(training_plan(method, config), ensure_ascii=False, indent=2))
            return 0
        validate_distributed_environment()
        validate_training_assets(method, config)
        # Only rank zero checks the shared destination before the native
        # process-group startup. Other ranks must not mistake files created
        # by a faster peer for an older run; torchrun propagates any failure.
        if os.environ["RANK"] == "0":
            validate_training_output(config)
    except (ValueError, FileNotFoundError) as exc:
        cli.error(str(exc))
    engine, _, _, trainer_module = engine_files(method)
    # Native engines have identically named Python packages; run one engine in
    # each torchrun process and resolve assets from the public release root.
    os.chdir(RELEASE_ROOT)
    sys.path.insert(0, str(engine))
    native = importlib.import_module(trainer_module)
    try:
        trainer = native.Trainer(config)
        initialize_smoke_ema(native, trainer, method, config)
        trainer.train()
    finally:
        # Import only after entering the actual training path.
        import torch.distributed as distributed
        if distributed.is_initialized():
            distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
