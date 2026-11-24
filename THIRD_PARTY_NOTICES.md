# Third-party notices and modification attribution

This release contains Hindsight Forcing and baseline implementations derived
from the projects below. Each project's existing copyrights and license terms
remain in effect. Public upstream project names, author notices, repository
URLs, and commit identifiers are intentionally retained for attribution.

## Component licenses

Revisions below identify the upstream source references, verified against the
retained source repositories and adapter provenance records. Exported engines
contain adaptations and are not byte-for-byte copies of those revisions.

| Component in this release | Upstream project | Upstream revision | License |
| --- | --- | --- | --- |
| `engines/hf` and the Causal Forcing-derived portions of the other engines | [Causal Forcing](https://github.com/thu-ml/Causal-Forcing) | `8202903297918bcd12810fc4230a4db7fe065e64` (reference upstream revision) | [Apache-2.0](licenses/Causal-Forcing-Apache-2.0.txt) |
| RAVEN-derived adapters, wherever included | [RAVEN](https://github.com/mvp-ai-lab/RAVEN) | `692ec211d4cd37a99a04773186fc2bf810430f78` | [CC BY-NC 4.0](licenses/RAVEN-CC-BY-NC-4.0.txt) |
| Reward Forcing adapters and reward-model wrapper | [Reward Forcing](https://github.com/JaydenLyh/Reward-Forcing) | `4317f5a0d29d1cee6a0f2e80ed5f5724bee97c66` | [Apache-2.0](licenses/Reward-Forcing-Apache-2.0.txt) |
| `engines/sgf` | [Self Gradient Forcing](https://github.com/zhuang2002/Self_Gradient_Forcing) | `79be9acbd874c9be050944a6b1de6693e988a171` | [Apache-2.0](licenses/SGF-Apache-2.0.txt) |
| `engines/sgf_plus` | [Self Gradient Forcing Plus](https://github.com/Zihan-Su/Self_Gradient_Forcing_Plus) | `7b313c8cbf42677ef55c8f4b05a258dd956ccb46` | [Apache-2.0](licenses/SGF-Plus-Apache-2.0.txt) |
| `engines/causvid`, including adapted inference code | [CausVid](https://github.com/tianweiy/CausVid) | `adb6a5ecd07666b4d0290042915c8406e6d5ce22` | [CC BY-NC-SA 4.0](licenses/CausVid-CC-BY-NC-SA-4.0.md) |

The RAVEN and CausVid licenses include noncommercial restrictions. CausVid's
license also includes share-alike conditions. The root Apache license does not
replace these terms. Apache notices attached to the shared Causal Forcing and
Wan components are retained even when those components occur in a differently
licensed adapter directory.

## Hindsight Forcing and common runtime changes

The Hindsight Forcing authors added the selected-step gradient graph,
full-rollout forward history, latent straight-through bridge, and differentiable
history writing to the Causal Forcing training implementation. The release
includes the main-paper training formulation and HF Long inference support.
Runtime adaptations include
checkpoint exports, accumulation and EMA handling, diagnostics, portable
configuration and entry points. Modified or adapted source files are identified
by their release modification notices and existing algorithm-specific headers.
These modifications do not imply endorsement by the upstream authors.

The `wan` source directories retain their existing Alibaba Wan Team copyright
notices. Wan model code, pretrained weights, tokenizers, and other downloaded
assets are distinct components; downloading them does not change their
upstream license terms.

## RAVEN adapter scope

The RAVEN distillation design was transcribed into the Causal Forcing engine.
The adapted parts include `model/raven_baseline.py`,
`pipeline/raven_training.py`, `wan/modules/raven_model.py`, and RAVEN-specific
logic in the shared baseline trainer and configuration. This file scope applies
to every engine containing those files, including copies under `engines/reward`.
The adapter preserves interleaved history encoding and the RAVEN distillation
objective; CM-GRPO is not part of this release's RAVEN baseline. Attribution to
RAVEN and its CC BY-NC 4.0 terms apply to these derived components.

## Reward Forcing adapter scope

The Reward Forcing adapter includes `model/reward_forcing_baseline.py`,
`pipeline/reward_forcing_training.py`, `utils/reward_forcing_reward.py`,
Reward Forcing-specific baseline trainer logic, and
`utils/reward_forcing_vendor/`. This applies wherever copies are present,
including `engines/raven`. The vendored reward class is adapted from
`videoalign/trainer.py:Qwen2VLRewardModelBT`; training-only imports were removed.
The prompt template originates from `videoalign/prompt_template.py`.
Any included attention reference is copied from the upstream
`CausalWanSelfAttention` implementation for validation.

The release retains Re-DMD reward weighting and EMA-Sink. The shared-runtime
adaptation includes gradient accumulation, checkpoint handling, and model
loading changes. The vendored license and source attribution accompany the
wrapper. The external VideoReward/Qwen model assets are not bundled.

## SGF and SGF+ adaptations

The two engines retain their separate upstream implementations. The matched
experiment adaptations concern common initialization, update-budget and EMA
handling, checkpoint and recovery handling, and runtime diagnostics. The SGF
two-pass history reconstruction and SGF+ writer/denoiser parameter separation
remain method-specific. Their source files remain under their respective
Apache licenses, including applicable upstream Wan and Causal Forcing notices.

## CausVid inference component

CausVid is included for inference with its official checkpoint. It is not a
retrained Hindsight Forcing model. Inference portability changes and any
packaging changes are modifications of the upstream component and retain its
CC BY-NC-SA 4.0 license. Upstream notices within its Wan implementation remain
in place.

## Checkpoints, datasets, and installed dependencies

Checkpoint catalogs distinguish official upstream releases from weights
trained for this work. A catalog entry is not a license grant for an upstream
checkpoint or dataset. Model files and training datasets are not embedded in
this source package. External packages installed from the requirement files
retain their own licenses. RollingForcing's separate `long_video` component
is not included in this release.
