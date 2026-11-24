#!/usr/bin/env python3
"""Check public package versions and optional imports without exposing local paths."""
from __future__ import annotations

import argparse
import contextlib
import importlib
from importlib import metadata
import io
import json
import platform
import warnings

# Distribution name, import name, required version (None: utility dependency).
CORE = (
    ("torch", "torch", "2.6.0"),
    ("torchvision", "torchvision", "0.21.0"),
    ("flash-attn", "flash_attn", "2.7.4.post1"),
    ("diffusers", "diffusers", "0.31.0"),
    ("transformers", "transformers", "4.49.0"),
    ("tokenizers", "tokenizers", None),
    ("numpy", "numpy", "1.24.4"),
    ("opencv-python", "cv2", "4.11.0.86"),
    ("av", "av", "13.1.0"),
    ("huggingface-hub", "huggingface_hub", "0.36.2"),
    ("accelerate", "accelerate", None),
    ("safetensors", "safetensors", None),
    ("Pillow", "PIL", None),
    ("omegaconf", "omegaconf", None),
    ("PyYAML", "yaml", None),
    ("einops", "einops", None),
    ("easydict", "easydict", None),
    ("ftfy", "ftfy", None),
    ("sentencepiece", "sentencepiece", None),
    ("tqdm", "tqdm", None),
    ("imageio", "imageio", None),
    ("imageio-ffmpeg", "imageio_ffmpeg", None),
    ("wandb", "wandb", None),
    ("tensorboard", "tensorboard", None),
    ("lmdb", "lmdb", None),
    ("packaging", "packaging", None),
    ("ninja", "ninja", None),
)
REWARD = (("peft", "peft", "0.14.0"),)


@contextlib.contextmanager
def quiet_import():
    # Dependency warnings/exceptions can contain installation paths. Only the
    # version and exception type are included in this public diagnostic output.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield


def inspect_environment(*, reward=False, imports=False, cuda=False):
    report = {"python": platform.python_version(), "packages": [], "ok": True}
    for distribution, module, required in CORE + (REWARD if reward else ()):
        row = {"package": distribution, "required": required or "installed"}
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            row.update(version=None, status="missing")
        else:
            row.update(version=version, status="ok")
            if required and version.split("+", 1)[0] != required:
                row["status"] = "version mismatch"
            if imports:
                try:
                    with quiet_import():
                        importlib.import_module(module)
                except Exception as error:
                    row["status"] = "import failed"
                    row["error_type"] = type(error).__name__
        report["packages"].append(row)
        if row["status"] != "ok":
            report["ok"] = False
    if cuda:
        try:
            with quiet_import():
                torch = importlib.import_module("torch")
                available = bool(torch.cuda.is_available())
                bf16 = bool(torch.cuda.is_bf16_supported()) if available else False
            report["cuda"] = {
                "build": torch.version.cuda,
                "available": available,
                "bf16_supported": bf16,
            }
            report["ok"] = report["ok"] and available and bf16
        except Exception as error:
            report["cuda"] = {"error_type": type(error).__name__}
            report["ok"] = False
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward", action="store_true", help="Also check Reward Forcing's PEFT dependency")
    parser.add_argument("--imports", action="store_true", help="Import packages to check native-extension compatibility")
    parser.add_argument("--cuda", action="store_true", help="Check CUDA and BF16 availability without loading models")
    parser.add_argument("--json", action="store_true", help="Print the same sanitized report as JSON")
    args = parser.parse_args(argv)
    report = inspect_environment(reward=args.reward, imports=args.imports, cuda=args.cuda)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("Python:", report["python"], "(reference: 3.11)")
        print(f"{'Package':<23} {'Installed':<22} Status")
        for row in report["packages"]:
            status = row["status"]
            if row.get("error_type"):
                status += " (" + row["error_type"] + ")"
            if status == "version mismatch":
                status += "; expected " + row["required"]
            print(f"{row['package']:<23} {row['version'] or '—':<22} {status}")
        if "cuda" in report:
            print("CUDA:", json.dumps(report["cuda"], sort_keys=True))
        print("Result:", "PASS" if report["ok"] else "CHECK REQUIRED")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
