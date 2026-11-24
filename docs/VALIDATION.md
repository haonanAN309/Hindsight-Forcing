# Release validation

Validation covers configuration, retained computation graphs, real-model inference, and training execution. Inference uses the public command-line entry. Training checks use the released trainers, through either the public entry or targeted native update calls. These are execution checks with real model assets, not reruns of the complete training budgets or benchmark cohorts.

## CPU and package checks

| Check | Result |
|---|---|
| Portable inference configuration tests | 6 passed |
| Portable training configuration and native trainer import tests | 15 passed |
| Retained HF computation-graph tests | 11 passed |
| Environment versions and actual package imports, including Reward Forcing dependencies | 28 passed |
| Native entry-point imports with SciPy and scikit-image unavailable | Passed |
| Python source syntax | Passed |
| Public checkpoint catalog against source file metadata | All 15 listed files identified |
| Uploaded EMA checkpoint integrity | Six full-file SHA-256 hashes verified |

The inference tests cover all nine methods at both supported lengths, explicit ordinary versus EMA weight selection, and FSDP wrapper-name normalization. Missing state-dictionary keys are rejected rather than replaced with another set of weights.

The training tests cover dynamic accumulation at global batch 64, method-specific update ratios and budgets, native SGF validators, import isolation, and the `--smoke-test` contract. Dry runs neither import PyTorch nor create output directories. Smoke checks use a separate output directory and reject an existing nonempty destination.

The HF graph tests exercise full rollout, the selected-step graph, differentiable history writing, future-loss feedback into the historical denoiser, detached critic sampling, and checkpoint recomputation. They use small CPU tensors rather than foundation-model weights.

## Real-model inference checks

Each method is checked with one short video and one one-minute video, using the same prompt and seed. Both lengths use the public `inference.py` entry, four denoising steps per chunk, `832 × 480` output resolution, and 16 FPS. Short videos have 81 decoded frames; long videos have 957 decoded frames.

Validation checks strict checkpoint loading, finite generated tensors, successful MP4 output, and complete video decoding with the expected frame count. SGF+ additionally checks its generation/memory expert routing. These are execution checks, not fresh quality estimates.

| Method | Short inference | One-minute inference |
|---|---|---|
| HF | Passed | Passed |
| HF Long | Passed | Passed |
| CausVid | Passed | Passed |
| Self Forcing | Passed | Passed |
| Causal Forcing | Passed | Passed |
| Reward Forcing | Passed | Passed |
| RAVEN | Passed | Passed |
| SGF | Passed | Passed |
| SGF+ | Passed | Passed |

All 18 inference cases passed, and the generated videos were fully decoded to verify the expected frames. Temporary test videos were removed after verification.

## Real-model training execution checks

Training checks exercise the released trainers with real model loading, rollout, losses, backward passes, and optimizer code. The scope is recorded per method; ordinary training configurations and budgets remain unchanged.

| Method | Verified execution | Checkpoint export | Status |
|---|---|---|---|
| HF | One student update and five critic updates, global batch 64 | Ordinary and EMA export passed; temporary weights removed | Passed |
| Reward Forcing | Student forward/backward with actual reward scoring on one micro-batch; critic forward/backward and optimizer update on one micro-batch | Not exercised | Passed for the listed scope |
| RAVEN, without CM-GRPO | One micro-batch critic update followed by one student update, with native rollout reuse and interleaved computation; EMA updated once | Not exercised | Passed for the listed scope |
| SGF | One micro-batch student forward/backward and optimizer update, plus one micro-batch critic forward/backward and optimizer update; EMA not exercised | Not exercised | Passed for the listed scope |
| SGF+ | One micro-batch student and critic optimizer update; EMA updated once; both generation and memory experts had nonzero optimizer moments, with no nonfinite DMD values | Not exercised | Passed for the listed scope |

Micro-batch checks validate actual computation without completing a full accumulated global-batch update. In the RAVEN check, losses and gradients were finite, nonzero Adam first moments confirmed updates to both networks, and EMA updated once; the teacher and text encoder stayed frozen. Checkpoint export is reported only where exercised.

### Optional complete update-cycle check

`--smoke-test` runs one student update and its associated critic updates using the real models, seven-chunk sequence, global batch 64, and original losses. It starts EMA immediately and exports ordinary/EMA weights into a separate `_smoke` directory. This optional check is broader than the targeted micro-batch checks above. Omitting the flag preserves the regular budget and EMA start.

## HF Long inference policy

HF Long is provided for inference only. Its short and long inference both use
`hf_long.pt`, a 12-latent-frame Sink+FIFO cache, contiguous RoPE positions, and
span 12. Both real-model inference checks passed; inference executes four
denoising steps per chunk and writes the final result.

## Reproduce the checks

Check installed packages and native imports without downloading assets:

```bash
python scripts/check_environment.py --imports
python scripts/check_environment.py --reward --imports
```

For CUDA and BF16 availability, add `--cuda`. The checker prints public versions and status without exposing local paths, hostnames, or device counts. The reported execution environment is not a claim that every dependency was reinstalled in a clean environment.

Run the CPU release checks from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  python -m unittest discover -s tests -v
```

Run the HF graph checks in a separate process to keep engine imports isolated:

```bash
cd engines/hf
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  python -m unittest discover -s tests -p test_ste_training.py -v
```

From the repository root, inspect or launch a real-model training check after preparing the assets:

```bash
python train.py --method hf --smoke-test --dry-run

torchrun --standalone --nproc_per_node=gpu train.py \
  --method hf --smoke-test \
  --data-path prompts/vidprom_filtered_extended.txt \
  --output-dir outputs/hf_check
```

The check writes to `outputs/hf_check_smoke/`. Change `--method` to another training entry to check that method. Use the ordinary inference examples in [README.md](../README.md) for full short/long video checks.

## Source and checkpoint provenance

The engines originate from the retained training snapshots and inference adapters used in the experiments. Release-specific entry points provide relative asset paths, derive accumulation from the current deployment, and add separate short-check controls. The SGF runtime also releases unused allocator cache once after initial mask/kernel setup, making space for communication buffers without changing the model, graph, masks, or training hyperparameters. Method-specific computation is documented in [METHODS.md](METHODS.md); upstream licenses and attribution remain in the release.

Model weights are downloaded separately, and their published SHA-256 hashes are recorded in the checkpoint catalog. Source-package privacy scanning checks for private paths, internal hostnames, credentials, and historical deployment topology while retaining public project links and author/license attribution.
