# Checkpoints

[model_index.json](model_index.json) records the weights used for the paper and the portable filenames expected by this release. Paths are relative to the repository root. This catalog contains nine method variants; the weight files themselves are not bundled.

Our retrained methods use their EMA weights after 150 generator updates. CausVid, Self Forcing, and Causal Forcing use the official released weights listed below. An official checkpoint's stored key is preserved: `generator` does not imply that the release also provides a separate EMA checkpoint.

| Method | Weight source | Evaluation file | Stored key | Generator updates | Global step |
|---|---|---|---|---:|---:|
| Hindsight Forcing | Our retraining | `checkpoints/hf.pt` | `generator_ema` | 150 | 750 |
| Hindsight Forcing Long | Our retraining | `checkpoints/hf_long.pt` | `generator_ema` | 150 | 750 |
| CausVid | Official four-step CFG-2 checkpoint | `checkpoints/causvid.pt` | `generator` | — | — |
| Self Forcing | Official DMD EMA checkpoint | `checkpoints/self_forcing.pt` | `generator_ema` | — | — |
| Causal Forcing | Official chunkwise checkpoint | `checkpoints/causal_forcing.pt` | `generator` | — | — |
| Reward Forcing | Our retraining | `checkpoints/reward_forcing.pt` | `generator_ema` | 150 | 750 |
| RAVEN, without CM-GRPO | Our retraining | `checkpoints/raven.pt` | `generator_ema` | 150 | 300 |
| SGF | Our retraining | `checkpoints/sgf.pt` | `generator_ema` | 150 | 750 |
| SGF+ | Our retraining | `checkpoints/sgf_plus.pt` | `generator_ema` | 150 | 750 |

RAVEN uses its method-specific update ratio, so 300 global steps correspond to the same 150 generator updates. Its checkpoint is not a 150-global-step checkpoint.

## Downloads and filenames

The six retrained EMA checkpoints are available in [NathanAN123/Hindsight_Forcing](https://huggingface.co/NathanAN123/Hindsight_Forcing), using flat filenames (`hf.pt`, `hf_long.pt`, `reward_forcing.pt`, `raven.pt`, `sgf.pt`, `sgf_plus.pt`). Access requires repository permission while it remains private. Only EMA weights are uploaded; ordinary weights remain local. Official releases are linked separately:

- [CausVid four-step CFG-2 weights](https://huggingface.co/tianweiy/CausVid/tree/main/autoregressive_checkpoint_warp_4step_cfg2)
- [Self Forcing DMD weights](https://huggingface.co/gdhe17/Self-Forcing/blob/main/checkpoints/self_forcing_dmd.pt)
- [Causal Forcing chunkwise weights](https://huggingface.co/zhuhz22/Causal-Forcing/blob/main/chunkwise/causal_forcing.pt)

Download the indicated file and place it at its catalog path. For example, the original `self_forcing_dmd.pt` file is assigned the portable local filename `checkpoints/self_forcing.pt`.

The common initialization for the retrained methods is [CF S2 `causal_ode.pt`](https://huggingface.co/zhuhz22/Causal-Forcing/blob/main/chunkwise/causal_ode.pt), assigned to `checkpoints/initialization/causal_ode.pt`. It is an initialization checkpoint, not the final Causal Forcing S3 baseline.

## Ordinary and EMA weights

The six retrained variants also have local, unuploaded ordinary student weights at `checkpoints/<method>/model.pt`, stored under `generator`. Both ordinary and EMA files are listed with their exact byte sizes in the catalog. These are model weight files; a full training-state resume also requires the relevant optimizer, scheduler, and random-state information.

Each ordinary or EMA file is approximately 5.29 GiB, except SGF+, whose files are approximately 10.57 GiB. Byte sizes are file-identification metadata, not checksums.

## Loading parameter names

Some EMA exports contain FSDP wrapper components in their parameter names. Normalize `_fsdp_wrapped_module.` before strict loading, retaining the `model.` prefix when loading the corresponding model wrapper. For example:

```python
from collections import OrderedDict
import torch

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
state = checkpoint[state_dict_key]  # Use the key recorded in model_index.json.
normalized = OrderedDict()
for name, tensor in state.items():
    name = name.replace("_fsdp_wrapped_module.", "")
    if name in normalized:
        raise ValueError(f"Duplicate parameter after normalization: {name}")
    normalized[name] = tensor

model_wrapper.load_state_dict(normalized, strict=True)
```

`model_wrapper` must have the architecture corresponding to the selected method. HF, Reward Forcing, and RAVEN EMA files require the wrapper-name normalization. The catalog records this property for each file.

## SGF+ parameter structure

Each SGF+ file contains both the generation/denoising parameters and the independently trained memory/context-writing parameters. The two parameter sets are saved together in one state dictionary. Use the SGF+ architecture, preserve all `memory_*` entries, and load strictly; dropping those entries changes the evaluated model.

## Hindsight Forcing variants

The main Hindsight Forcing checkpoint is the paper's short-video model: selected steps are synchronized across ranks, each chunk completes the full four-step rollout, full-rollout outputs supply forward history, and the latent bridge uses scale one. Its original short-video configuration uses a 21-latent-frame cache.

Hindsight Forcing Long is a long-horizon model trained with **per-rank independent step sampling**. It is available for inference only via `python inference.py --method hf_long`. Its short- and long-video evaluation uses a 12-latent-frame cache with contiguous RoPE positions and span 12. The default checkpoint is `checkpoints/hf_long.pt`.

## Download our EMA weights

```bash
hf download NathanAN123/Hindsight_Forcing hf.pt --local-dir checkpoints
hf download NathanAN123/Hindsight_Forcing hf_long.pt --local-dir checkpoints
hf download NathanAN123/Hindsight_Forcing reward_forcing.pt --local-dir checkpoints
hf download NathanAN123/Hindsight_Forcing raven.pt --local-dir checkpoints
hf download NathanAN123/Hindsight_Forcing sgf.pt --local-dir checkpoints
hf download NathanAN123/Hindsight_Forcing sgf_plus.pt --local-dir checkpoints
```
