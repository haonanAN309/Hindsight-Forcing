# Modified for the Hindsight Forcing release.
# See the repository-root THIRD_PARTY_NOTICES.md for provenance and component licenses.
"""Frozen VideoAlign MQ inference with its trained RGB input convention."""
import json
from pathlib import Path
import sys
import time
import torch


def prepare_reward_videos(pixels):
    """Convert Wan's [-1,1] output to VideoReward's RGB [0,255] input.

    The released Reward Forcing adapter multiplies signed Wan pixels by 255,
    unlike VideoReward's real-video training input.  That both changes color
    semantics and breaks Qwen's PIL resize.  Restore RGB first, bicubic resize
    in float32, then clamp interpolation overshoot.  Qwen must skip its second
    resize because 336x588 is already a multiple of its patch/merge factor 28.
    Keep all frames and the original MQ normalization / Re-DMD weighting.
    """
    from torchvision.transforms import functional as TF, InterpolationMode
    if pixels.ndim != 5 or pixels.shape[2] != 3:
        raise ValueError("Reward pixels must have shape [B,T,3,H,W]")
    if not torch.isfinite(pixels).all():
        raise FloatingPointError("Non-finite decoded pixels before reward preprocessing")
    return [TF.resize((sample.detach().float() + 1.0) * 127.5, [336, 588],
                      interpolation=InterpolationMode.BICUBIC, antialias=True).clamp_(0, 255).cpu()
            for sample in pixels]


class VideoRewardAdapter:
    def __init__(self, reward_path, base_path, dependency_path=None, offload=True,
                 attention_backend="flash_attention_2"):
        if dependency_path and str(dependency_path) not in sys.path:
            sys.path.insert(0, str(dependency_path))
        from peft import LoraConfig, get_peft_model
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoProcessor
        from utils.reward_forcing_vendor.qwen_reward import Qwen2VLRewardModelBT
        reward_path, base_path = Path(reward_path), Path(base_path)
        metadata = json.loads((reward_path / "model_config.json").read_text())
        state = torch.load(reward_path / "checkpoint-11352/model.pth", map_location="cpu", weights_only=True, mmap=True)
        config = AutoConfig.from_pretrained(base_path, local_files_only=True)
        config.vocab_size = state["base_model.model.model.embed_tokens.weight"].shape[0]
        config._attn_implementation = attention_backend
        self.processor = AutoProcessor.from_pretrained(base_path, padding_side="right", local_files_only=True)
        self.processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>"]})
        special_ids = self.processor.tokenizer.convert_tokens_to_ids(["<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>"])
        with init_empty_weights(include_buffers=False):
            model = Qwen2VLRewardModelBT(config, output_dim=1, reward_token="special", special_token_ids=special_ids)
        targets = [name for name, module in model.named_modules()
                   if isinstance(module, torch.nn.Linear)
                   and not any(skip in name for skip in ("lm_head", "rm_head", "embed_tokens", "visual"))]
        lora = metadata["peft_lora_config"]
        model = get_peft_model(model, LoraConfig(
            target_modules=targets, r=lora["lora_r"], lora_alpha=lora["lora_alpha"],
            lora_dropout=lora["lora_dropout"], task_type="CAUSAL_LM",
            use_rslora=lora["use_rslora"], bias="none"))
        # The pinned transformers 4.49 layout matches the released model.pth.
        # Strict loading prevents silent loss of the trained reward head/LoRA.
        model.load_state_dict(state, strict=True, assign=True)
        self.model = model.eval().requires_grad_(False)
        self.model = self.model.to(dtype=torch.bfloat16)
        self.norm = metadata["inference_config"]
        self.offload = offload
        self.last_metrics = {}

    @torch.no_grad()
    def __call__(self, latents, prompts, vae):
        from utils.reward_forcing_vendor.prompt_template import build_prompt
        device = latents.device
        decode_start = time.monotonic()
        vae.to(device=device, dtype=torch.bfloat16)
        pixels = vae.decode_to_pixel(latents.detach()).to(latents.dtype)
        if self.offload:
            vae.to("cpu")
        videos = prepare_reward_videos(pixels)
        del pixels
        decode_secs = time.monotonic() - decode_start
        reward_start = time.monotonic()
        chat = [[{"role": "user", "content": [
            {"type": "video", "video": "file://dummy_path"},
            {"type": "text", "text": build_prompt(prompt, ["VQ", "MQ", "TA"], "detailed_special")},
        ]}] for prompt in prompts]
        inputs = self.processor(
            text=self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True),
            images=None, videos=videos, padding=True, return_tensors="pt",
            videos_kwargs={"do_rescale": True, "do_resize": False})
        self.model.to(device)
        inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
        try:
            logits = self.model(**inputs, return_dict=True)["logits"].float()
            mq = ((logits[:, 1] - self.norm["MQ_mean"]) / self.norm["MQ_std"]).detach()
        finally:
            if self.offload:
                self.model.to("cpu")
        self.last_metrics = {"reward_decode_seconds": decode_secs,
                             "reward_model_seconds": time.monotonic() - reward_start}
        return mq
