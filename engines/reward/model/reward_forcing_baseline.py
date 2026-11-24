# Modified for the Hindsight Forcing release.
# See the repository-root THIRD_PARTY_NOTICES.md for provenance and component licenses.
"""Matched-budget Reward Forcing: Re-DMD plus the released EMA-Sink algorithm."""
import torch
from model.dmd import DMD
from pipeline.reward_forcing_training import RewardForcingTrainingPipeline, install_reward_attention


def rewarded_dmd_loss(prediction, gradient, reward_mq, reward_inverse_temperature=2.0, mask=None):
    """Reward is a detached sample weight, NOT a differentiable auxiliary loss."""
    error = (prediction.double() - (prediction.double() - gradient.double()).detach()).square()
    if mask is None:
        per_sample = error.flatten(1).mean(1)
    else:
        counts = mask.flatten(1).sum(1)
        if (counts == 0).any():
            raise ValueError("Reward DMD received a sample with no supervised elements")
        per_sample = (error * mask).flatten(1).sum(1) / counts
    weights = torch.exp(float(reward_inverse_temperature) * reward_mq.detach().float())
    return 0.5 * (weights * per_sample).mean(), weights.detach()


class RewardForcingBaseline(DMD):
    def __init__(self, args, device):
        super().__init__(args, device)
        if getattr(args, "ste_training", False):
            raise ValueError("Reward Forcing retains its own early-exit rollout; STE must be off")
        install_reward_attention(self.generator.model)
        self.reward_adapter = None

    def prepare_auxiliary(self):
        """CPU-only initialization; safe after backbone FSDP setup."""
        if self.reward_adapter is None:
            from utils.reward_forcing_reward import VideoRewardAdapter
            self.reward_adapter = VideoRewardAdapter(
                self.args.reward_model_path, self.args.reward_base_model_path,
                getattr(self.args, "reward_dependency_path", None),
                offload=getattr(self.args, "reward_cpu_offload", True))

    def _initialize_inference_pipeline(self):
        self.inference_pipeline = RewardForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list, scheduler=self.scheduler,
            generator=self.generator, num_frame_per_block=self.num_frame_per_block,
            same_step_across_blocks=self.same_step_across_blocks,
            last_step_only=self.args.last_step_only, context_noise=self.args.context_noise,
            local_attn_size=self.args.model_kwargs.local_attn_size,
            sink_size=self.args.model_kwargs.sink_size,
            sink_decay=getattr(self.args, "reward_sink_decay", 0.999),
            cache_frames=getattr(self.args, "reward_training_cache_frames", 21))

    def generator_loss(self, image_or_video_shape, conditional_dict, unconditional_dict,
                       clean_latent=None, initial_latent=None, text_prompts=None):
        if text_prompts is None:
            raise ValueError("Reward Forcing needs the actual generation prompts")
        prediction, mask, from_t, to_t = self._run_generator(
            image_or_video_shape=image_or_video_shape, conditional_dict=conditional_dict,
            clean_latent=clean_latent, initial_latent=initial_latent)
        self.prepare_auxiliary()
        reward_mq = self.reward_adapter(prediction.detach(), text_prompts, self.vae)
        grad, logs = self._sample_distribution_matching_gradient(
            prediction, conditional_dict, unconditional_dict, from_t, to_t)
        loss, weights = rewarded_dmd_loss(
            prediction, grad, reward_mq,
            getattr(self.args, "reward_inverse_temperature", 2.0), mask)
        logs.update({"reward_mq": reward_mq.mean(), "reward_weight": weights.mean(),
                     "reward_weight_max": weights.max()})
        logs.update({key: prediction.new_tensor(value) for key, value in
                     {**self.reward_adapter.last_metrics, **self.inference_pipeline.last_runtime_stats}.items()})
        return loss, logs
