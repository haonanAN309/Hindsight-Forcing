# Methods and matched training settings

This document describes the configurations shipped in `engines/`. Hindsight Forcing (HF) means the main paper's short-video method. Hindsight Forcing Long is provided for inference only.

## Shared experiment settings

The retrained methods initialize the student from the same Causal Forcing S2 causal ODE weights. They use Wan2.1-T2V-1.3B as the student and trainable diffusion critic, a frozen Wan2.1-T2V-14B teacher, and the same filtered and extended VidProM prompts. The text encoder and VAE are frozen. Videos are generated online from prompts.

Training samples contain 21 latent frames, divided into seven chunks of three frames. The latent spatial shape is `16 × 60 × 104` per frame, corresponding to the evaluated `832 × 480` video resolution. Each optimizer update accumulates a global batch of 64. BF16 computation, FSDP, activation checkpointing, and the recorded random seed are retained. The runtime determines accumulation from the user's deployment; the release does not prescribe the original cluster topology.

All retrained methods have 150 actual student optimizer updates. A global step includes one critic update and, according to the method's ratio, a student update. Thus matching generator updates does not require the same number of global steps.

| Method | Student updates | Critic updates | Global steps | Critic : student ratio | Critic-only warmup |
|---|---:|---:|---:|---:|---:|
| HF | 150 | 750 | 750 | 5 : 1 | 0 |
| RAVEN without CM-GRPO | 150 | 300 | 300 | 2 : 1 | 0 |
| Reward Forcing | 150 | 750 | 750 | 5 : 1 | 0 |
| SGF | 150 | 750 | 750 | 5 : 1 | 0 |
| SGF+ | 150 | 750 | 750 | 5 : 1 | 0 |

The optimizer is AdamW, with student learning rate `2e-6`, critic learning rate `4e-7`, and betas `(0, 0.999)`. RAVEN retains weight decay zero and its gradient clipping configuration; the other methods use the inherited weight decay `0.01`. Configured context noise is `context_noise: 0`. Method-specific schedules, attention construction, objectives, and gradient paths are preserved.

EMA decay is `0.99` and EMA updates occur only after student optimizer updates. HF initializes EMA at the end of global step 200, after 40 student updates. The other retrained methods explicitly initialize it at student update 40. The subsequent 110 student updates update the EMA, and the main evaluations use the EMA at student update 150. Ordinary and EMA student weights are both cataloged.

## Training-time history and gradient paths

Here, “future gradient” means feedback from a later chunk's loss through its use of an earlier chunk as context.

| Method | Selected-step sampling | Forward history source | Future gradient into history encoding | Future gradient into historical generation |
|---|---|---|---|---|
| HF | One selected step shared across chunks and synchronized across ranks | Final output of the complete four-step rollout | Yes, through the differentiable KV writer | Yes, through the latent straight-through bridge |
| RAVEN | A selected trajectory step sampled per example, shared across its chunks; no cross-rank synchronization | Detached complete-rollout endpoints | Yes, through the interleaved training pass | No |
| Reward Forcing | Selected step synchronized across ranks and shared across chunks | Detached selected-step clean estimate | No; KV writing is detached | No |
| SGF | Independent selected step per rank, shared across chunks on that rank | Detached selected-step clean estimate | Yes, through the second teacher-forced pass | No |
| SGF+ | Independent selected step per rank, shared across chunks on that rank | Detached selected-step clean estimate | Yes, through the separate memory expert in the second pass | No |

“Shared parameters” does not imply an undetached historical generation path. RAVEN and SGF can update parameters used for encoding history while keeping the generated history latent detached.

## Hindsight Forcing

Configuration: [`engines/hf/configs/hf.yaml`](../engines/hf/configs/hf.yaml).

Each chunk executes all four denoising calls. Only the randomly selected call retains its denoising graph; transitions between denoising steps are detached. Let `x` be the final rollout output and `z` the selected-step clean estimate. HF supplies

```text
history = stop_gradient(x) + alpha * (z - stop_gradient(z))
```

to the history-writing call, with `alpha = 1`. The forward history therefore equals the complete-rollout output. The writer and committed KV tensors retain their graphs, so a later chunk's loss can update both history encoding and the selected historical denoiser. The selected clean estimates, concatenated across chunks, receive the original DMD supervision. Teacher and critic predictions provide the detached DMD direction; the critic is updated separately with the flow denoising objective.

HF adds no separate writer parameters, reward objective, GAN branch, or regression loss. Its four-step schedule uses the CF index list `[1000, 750, 500, 250]` with schedule warping enabled and shift `5`. The main training retains the available sequence history rather than introducing a local-history training window.

Implementation: [`pipeline/ste_training.py`](../engines/hf/pipeline/ste_training.py), [`model/dmd.py`](../engines/hf/model/dmd.py), and [`trainer/distillation.py`](../engines/hf/trainer/distillation.py) under `engines/hf/`.

## RAVEN without CM-GRPO

Configuration: [`engines/raven/configs/raven.yaml`](../engines/raven/configs/raven.yaml).

The CF-based adapter retains RAVEN's detached complete rollout and trajectory reuse. It interleaves noisy chunks with detached clean history, duplicates their original temporal positions, and evaluates the selected denoising step in a differentiable parallel pass. Later chunks can train the history encoding, while the rollout that produced the clean history remains detached.

The adapter preserves the four-step integer schedule `[1000, 938, 833, 625]`, FP32 trajectory state with BF16 model computation, critic-first ordering, two critic updates per student update, teacher CFG, and the native score-noise sampler. It retains normalized reverse-participation chunk weights: for `N` chunks the weights are proportional to `1 / [N, N-1, ..., 1]`, favoring later chunks. CM-GRPO is not enabled.

Implementation: [`pipeline/raven_training.py`](../engines/raven/pipeline/raven_training.py), [`model/raven_baseline.py`](../engines/raven/model/raven_baseline.py), [`wan/modules/raven_model.py`](../engines/raven/wan/modules/raven_model.py), and [`trainer/baseline_distillation.py`](../engines/raven/trainer/baseline_distillation.py).

## Reward Forcing

Configuration: [`engines/reward/configs/reward.yaml`](../engines/reward/configs/reward.yaml).

The adapter preserves Re-DMD and EMA-Sink. Training exits each chunk at the selected denoising step and writes a detached selected-step prediction into history. A frozen VideoReward model evaluates the decoded video. Its normalized motion-quality reward supplies a detached per-sample DMD multiplier `exp(2 * reward_MQ)`. The reward is a sample weight, not a differentiable loss through the reward model.

EMA-Sink stores unrotated keys and reapplies RoPE when reading a packed window. The training buffer has 21 latent frames, while attention reads the configured nine-frame span. The inference buffer has nine latent frames. The three-frame sink is updated with decay `0.999` when a buffer eviction occurs. This preserves the distinction between buffer capacity and attended context.

The released adapter also retains the verified pixel-input conversion: decoded Wan pixels in `[-1, 1]` become RGB values in `[0, 255]`, are resized to `336 × 588`, and are clamped after interpolation before the reward processor. The reward model and VAE are offloaded according to the configuration. Reward evaluation is required for training, not for student inference.

Implementation: [`model/reward_forcing_baseline.py`](../engines/reward/model/reward_forcing_baseline.py), [`pipeline/reward_forcing_training.py`](../engines/reward/pipeline/reward_forcing_training.py), and [`utils/reward_forcing_reward.py`](../engines/reward/utils/reward_forcing_reward.py).

## SGF and SGF+

Configurations: [`engines/sgf/configs/sgf.yaml`](../engines/sgf/configs/sgf.yaml) and [`engines/sgf_plus/configs/sgf_plus.yaml`](../engines/sgf_plus/configs/sgf_plus.yaml).

These methods retain their separate native implementations, with a matched runtime for initialization, accumulation, update counting, checkpoint export, and monitoring. They are not replaced by HF's training pipeline.

Pass one performs a detached autoregressive rollout and records the noisy input and clean estimate at the selected step. The configurations use `per_rank_exit_step: true`, `self_gradient_forcing_cache_mode: exit`, and `self_gradient_forcing_match_context: true`. With these settings, the implementation executes all four denoising calls but commits the selected-step estimate to history. Therefore completing four calls does not mean the final-step output supplies history.

Pass two runs a differentiable parallel teacher-forced forward over the recorded noisy inputs and detached matched history. This restores gradients into history encoding, without restoring gradients through the earlier rollout that generated those history latents. Both methods retain `local_attn_size: 9` and `sink_size: 3` in their native attention implementation.

SGF shares context-writing and denoising parameters. SGF+ enables `dual_full_expert`, with independent memory/context-writing parameters and generation/denoising parameters. At initialization, the CF S2 parameters are copied into both matching branches; they then train independently. Both branches are saved in each SGF+ student or EMA file, and both must be loaded.

Implementation in each engine: `pipeline/self_gradient_forcing_training.py`, `wan/modules/causal_model.py`, `utils/wan_wrapper.py`, `model/dmd.py`, and `trainer/matched_runtime.py`.

## Official-checkpoint baselines

CausVid, Self Forcing, and Causal Forcing are evaluated using official released weights; this paper does not assign them the retraining counts above. The release retains their evaluation adapters and CausVid's native inference implementation. Refer to [`checkpoints/model_index.json`](../checkpoints/model_index.json) for the exact release files and state keys.

## Hindsight Forcing Long

HF Long is a long-horizon model trained with **per-rank independent step sampling**.
It is provided for inference only through `python inference.py --method hf_long`.
It loads `hf_long.pt` through the SGF-derived rolling-cache backend, with a
12-latent-frame Sink+FIFO budget, a 3-frame sink, contiguous RoPE positions,
and span 12. The model attention configuration matches evaluation
(`local_attn_size=9`, `sink_size=3`). Both short and long inference execute
four denoising steps per chunk and commit the final output.
See [`configs/inference.json`](../configs/inference.json) for the inference configuration.
