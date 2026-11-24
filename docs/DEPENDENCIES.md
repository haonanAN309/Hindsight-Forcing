# Dependencies and model preparation

## Minimal environment checklist

Use **Linux, Python 3.11, and a CUDA-capable NVIDIA GPU**. The reference software stack is:

| Layer | Version / requirement | Role |
|---|---|---|
| Python | 3.11 | Runtime |
| PyTorch | 2.6.0, CUDA 12.4 build | BF16, FSDP, FlexAttention, checkpointing |
| torchvision | 0.21.0 | Video and image utilities |
| FlashAttention | 2.7.4.post1 | Attention kernels |
| diffusers | 0.31.0 | Wan configuration and schedulers |
| transformers | 4.49.0 | UMT5 tokenizer and Qwen2-VL reward model |
| NumPy / PyAV | 1.24.4 / 13.1.0 | Array operations and video encoding |
| Hugging Face Hub | 0.36.2 | Download CLI and model loading |
| OmegaConf / PyYAML | See `requirements.txt` | Configuration loading |
| PEFT | 0.14.0 | Reward Forcing's LoRA-based reward model |
| Build utilities | `packaging`, `ninja`, a C++ compiler, CUDA toolkit when building kernels | FlashAttention installation |

The root [`requirements.txt`](../requirements.txt) contains the shared training/inference dependencies, including small utilities imported by the native engines. Optional browser demos, ONNX export, external prompt expansion, and unrelated inherited benchmarks are outside these entry points and may require their upstream extras. `scikit-image` is listed in inherited engine requirements but is not imported by the released entry points; SciPy is optional in the inherited scheduler modules and is not needed by the released four-step inference paths. Neither is part of the minimal root requirements. Direct imports of `regex` and `requests` are satisfied by the pinned Transformers/Hugging Face dependency chains.

### Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Install a driver compatible with the CUDA build. FlashAttention is installed after PyTorch because its extension must match PyTorch and CUDA; compiling from source additionally requires a CUDA toolkit and compiler. The requirements pin the model-facing libraries; utility packages without exact pins are not a full environment lockfile.

### Check the environment

```bash
python -m pip check
python scripts/check_environment.py --imports --cuda
# Also check the extra dependency used for Reward Forcing training:
python scripts/check_environment.py --reward --imports
python train.py --method hf --dry-run
python inference.py --method hf --prompt-file examples/prompts.txt --dry-run
```

The checker reports installed versions, missing packages, native import errors, and optional CUDA/BF16 availability. It prints no local paths, hostnames, or device counts. `--json` produces a machine-readable report. It checks the installed environment; it does not install dependencies or download models.

Dry runs verify configuration and plan construction; [VALIDATION.md](VALIDATION.md) records the separate execution checks. GPU memory requirements depend on the selected method, sequence length, micro-batch, and sharding configuration. The training launcher derives accumulation from the current distributed world size while preserving global batch 64.

### Required assets by task

| Task | Required assets |
|---|---|
| Generate with any released student | Complete Wan2.1-T2V-1.3B directory + that student's checkpoint |
| HF / RAVEN / SGF / SGF+ training | Above foundation assets + Wan2.1-T2V-14B teacher + CF S2 initialization + training prompt text file |
| Reward Forcing training | Same training assets + VideoReward + Qwen2-VL configuration/processor/tokenizer |
| CPU configuration checks | Python dependencies; no model downloads |

Generation uses supplied text prompts and does not call an online prompt-expansion API. Ordinary and EMA student files do not contain the frozen text encoder or VAE, so the foundation assets remain necessary even after downloading a trained student.

## Student, teacher, text encoder, and VAE

Prepare the upstream Wan model directories under the repository root:

```text
wan_models/
  Wan2.1-T2V-1.3B/
  Wan2.1-T2V-14B/
```

The 1.3B directory supplies the base student/critic architecture and weights, the frozen UMT5 text encoder and tokenizer, and the Wan VAE. The 14B directory supplies the frozen bidirectional teacher for training. The wrappers expect the original Wan directory structure, including:

```text
wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth
wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/
wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
```

Use the upstream model releases [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) and [Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B). Student-only inference uses the selected method's student checkpoint plus the corresponding text encoder/VAE assets; it does not load a DMD teacher or critic.

The common student initialization is CF S2 `causal_ode.pt`. The checkpoint catalog assigns it to:

```text
checkpoints/initialization/causal_ode.pt
```

The supplied launcher and training configurations use this path. The S2 initialization and the final CF S3 evaluation checkpoint are different weights.

[`checkpoints/model_index.json`](../checkpoints/model_index.json) specifies evaluation weights, exact expected filenames, and state-dictionary keys. The six retrained EMA checkpoints are hosted at [NathanAN123/Hindsight_Forcing](https://huggingface.co/NathanAN123/Hindsight_Forcing); download links and SHA-256 hashes are recorded in the catalog. [`checkpoints/README.md`](../checkpoints/README.md) explains ordinary versus EMA files, FSDP prefix normalization, and SGF+'s dual parameter sets.

## Training prompts

The matched retraining configurations read a UTF-8 text file at:

```text
prompts/vidprom_filtered_extended.txt
```

Use the filtered and extended VidProM prompt list corresponding to the CF setup. Training samples videos online from those prompts; it does not require paired clean-video latents for HF, RAVEN, Reward Forcing, SGF, or SGF+. CausVid is an official-checkpoint baseline in this release, so its paired-data training preparation is not part of these matched retraining instructions.

## Additional Reward Forcing assets

Reward Forcing training additionally loads the frozen [VideoReward](https://huggingface.co/KlingTeam/VideoReward) model and the [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) processor/base configuration. Prepare:

```text
checkpoints/rewards/
  VideoReward/
    model_config.json
    checkpoint-11352/
      model.pth
  Qwen2-VL-2B-Instruct/
    config.json
    # Keep all processor and tokenizer files from the upstream release.
```

The corresponding configuration keys are `reward_model_path` and `reward_base_model_path`. The loader uses local files, reconstructs the LoRA and reward-head structure from `model_config.json`, and strictly loads the released `model.pth`. The VideoReward file includes the trained state; the Qwen directory provides the model configuration, processor, and tokenizer used to interpret it.

The adapter requires `peft==0.14.0`, `transformers==4.49.0`, `accelerate`, torchvision, and FlashAttention 2. Its Qwen reward-model and prompt-template definitions are retained under `engines/reward/utils/reward_forcing_vendor/`. External `qwen_vl_utils` is not imported by this reward adapter. `reward_dependency_path` can remain empty when PEFT is installed in the normal Python environment.

During training, the VAE decodes generated latents for reward scoring, and the detached motion-quality score reweights DMD. The configured reward CPU offload remains available. The reward model is not required to generate videos from the trained Reward Forcing student.

## Engine isolation

The native engines contain identically named Python packages such as `model`, `pipeline`, `utils`, and `wan`. Launch a method through its isolated release entry point; do not combine all engine directories in one `PYTHONPATH` or import several of them into the same interpreter. Shared model assets can remain in the repository-root directories above.
