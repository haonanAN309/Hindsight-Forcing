# Modified for the Hindsight Forcing release.
# See the repository-root THIRD_PARTY_NOTICES.md for provenance and component licenses.
"""CPU contracts for the opt-in STE rollout; no model weights or GPU required.

Run with ``python -m unittest discover -s tests -p test_ste_training.py``.
Only heavyweight import dependencies are omitted when loading production code.
The rollout, critic loss, functional attention and checkpoint helper under test
are the actual source from this checkout.
"""

import ast
import math
from pathlib import Path
import types
from typing import List, Optional
import unittest
from unittest import mock

import torch
from torch import nn
import torch.utils.checkpoint


ROOT = Path(__file__).resolve().parents[1]


def load_source(relative_path, namespace, excluded_imports=()):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module in excluded_imports
        )
    ]
    namespace = {"__name__": f"ste_test_{path.stem}", **namespace}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


baseline_module = load_source(
    "pipeline/self_forcing_training.py",
    {"WanDiffusionWrapper": object, "SchedulerInterface": object},
    ("utils.wan_wrapper", "utils.scheduler"),
)
BaselinePipeline = baseline_module["SelfForcingTrainingPipeline"]
ste_module = load_source(
    "pipeline/ste_training.py",
    {"SelfForcingTrainingPipeline": BaselinePipeline},
    ("pipeline", "pipeline.self_forcing_training", "self_forcing_training"),
)
STEPipeline = ste_module["STETrainingPipeline"]
dmd_module = load_source(
    "model/dmd.py",
    {"SelfForcingModel": nn.Module, "SelfForcingTrainingPipeline": BaselinePipeline},
    ("pipeline", "model.base"),
)
DMD = dmd_module["DMD"]


def cpu_attention(query, key, value):
    return torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    ).transpose(1, 2)


class SmallCrossAttention(nn.Module):
    """CPU context projection for exercising the real Wan block plumbing."""

    def __init__(self, dim, *args):
        super().__init__()
        self.projection = nn.Linear(dim, dim)

    def forward(self, hidden, context, context_lens, crossattn_cache=None):
        if crossattn_cache is not None:
            raise AssertionError("STE must recompute cross-attention projections")
        return self.projection(context).mean(dim=1, keepdim=True).expand_as(hidden)


def load_cache_contracts():
    """Use real cache/attention code with CPU SDPA instead of FlashAttention."""
    path = ROOT / "wan/modules/causal_model.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    names = {
        "_ste_cached_attention", "_forward_ste_block",
        "causal_rope_apply", "CausalWanSelfAttention", "CausalWanAttentionBlock",
        "CausalHead", "CausalWanModel",
    }
    tree.body = [node for node in tree.body if getattr(node, "name", None) in names]
    embedding_source = ast.parse((ROOT / "wan/modules/model.py").read_text())
    embedding = next(node for node in embedding_source.body if getattr(node, "name", None) == "sinusoidal_embedding_1d")
    tree.body.insert(0, embedding)
    namespace = {
        "torch": torch, "nn": nn, "math": math, "attention": cpu_attention,
        "WanLayerNorm": nn.LayerNorm, "ModelMixin": nn.Module,
        "ConfigMixin": type("ConfigMixin", (), {}), "BlockMask": object,
        "register_to_config": lambda function: function,
        "WAN_CROSSATTENTION_CLASSES": {"test": SmallCrossAttention},
    }
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


cache_module = load_cache_contracts()
cached_attention = cache_module["_ste_cached_attention"]
forward_ste_block = cache_module["_forward_ste_block"]
CausalAttention = cache_module["CausalWanSelfAttention"]
CausalBlock = cache_module["CausalWanAttentionBlock"]
CausalModel = cache_module["CausalWanModel"]
CausalHead = cache_module["CausalHead"]


def load_wan_wrapper():
    path = ROOT / "utils/wan_wrapper.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [node for node in tree.body if getattr(node, "name", None) == "WanDiffusionWrapper"]
    namespace = {
        "torch": torch, "nn": nn, "List": List, "Optional": Optional,
        "SchedulerInterface": object,
    }
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["WanDiffusionWrapper"]


WanWrapper = load_wan_wrapper()


class ToyScheduler:
    """A deterministic formula with genuine stochastic renoising inputs."""

    def __init__(self):
        self.timesteps = torch.arange(1000, 0, -1)
        self.sigmas = self.timesteps / 1000
        self.alphas_cumprod = None

    def add_noise(self, clean, noise, timestep):
        fraction = timestep.flatten().to(clean) / 10000
        fraction = fraction.reshape(-1, *([1] * (clean.ndim - 1)))
        return clean * (1 - fraction) + noise * fraction

    def convert_x0_to_noise(self, x0, xt, timestep):
        return xt - x0


class ToyStudent(nn.Module):
    """Reads committed history and exposes every per-call autograd boundary."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
        self.writer_weight = nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
        self.calls = []

    def forward(
        self,
        noisy_image_or_video,
        conditional_dict,
        timestep,
        kv_cache,
        crossattn_cache,
        current_start,
        **kwargs,
    ):
        latent = noisy_image_or_video
        cache = kv_cache[0]
        if cache.get("ste_cache", False):
            chunks = cache["ste_v_chunks"]
            if chunks:
                history = torch.cat(chunks, dim=1).sum(dim=(1, 2, 3))
            else:
                history = latent.new_zeros(latent.shape[0])
        else:
            history = cache["v"][:, :current_start].sum(dim=(1, 2, 3))
        history = history.reshape(-1, 1, 1, 1, 1)
        time = timestep.to(latent).reshape(*timestep.shape, 1, 1, 1)
        prediction = 0.3 * latent + self.weight * (1 + time / 1000) + 0.2 * history
        value = latent.reshape(latent.shape[0], latent.shape[1], 1, 1) * self.writer_weight
        if not cache.get("ste_cache", False):
            # The original pipeline uses mutable slots. This toy's copies make
            # the baseline test independent of Wan's in-place attention kernel.
            updated = cache["v"].clone()
            updated[:, current_start:current_start + value.shape[1]] = value
            cache["v"] = updated
        if prediction.requires_grad:
            prediction.retain_grad()
        if latent.requires_grad:
            latent.retain_grad()
        self.calls.append(
            {
                "timestep": int(timestep[0, 0]),
                "start": current_start,
                "grad_enabled": torch.is_grad_enabled(),
                "input": latent,
                "output": prediction,
                "history": history,
            }
        )
        if cache.get("ste_cache", False) and "ste_cache_output" in cache:
            # The pipeline must consume public writer outputs, as FSDP installs
            # backward hooks on returned tensors. Deliberately leave the cache
            # side-channel empty so these tests reject a side-effect-only path.
            return prediction * 0, prediction, ((value + 0.1, value),)
        return prediction * 0, prediction


def initialize_small_cross_cache(self, batch_size, dtype, device):
    self.crossattn_cache = [
        {
            "k": torch.zeros(batch_size, 1, 1, 1, dtype=dtype, device=device),
            "v": torch.zeros(batch_size, 1, 1, 1, dtype=dtype, device=device),
            "is_init": False,
        }
    ]


def initialize_small_baseline_cache(self, batch_size, dtype, device):
    self.kv_cache1 = [
        {
            "k": torch.zeros(batch_size, self.kv_cache_size, 1, 1, dtype=dtype, device=device),
            "v": torch.zeros(batch_size, self.kv_cache_size, 1, 1, dtype=dtype, device=device),
            "global_end_index": torch.tensor([0]),
            "local_end_index": torch.tensor([0]),
        }
    ]


def make_pipeline(selected=1, alpha=1.0, history_grad=True, baseline=False):
    student = ToyStudent()
    pipeline_type = BaselinePipeline if baseline else STEPipeline
    extra = {} if baseline else {"ste_alpha": alpha, "ste_history_grad": history_grad}
    pipeline = pipeline_type(
        denoising_step_list=torch.tensor([900, 700, 400, 100]),
        scheduler=ToyScheduler(),
        generator=student,
        num_frame_per_block=1,
        same_step_across_blocks=not baseline,
        num_max_frames=8,
        context_noise=0,
        **extra,
    )
    pipeline.num_transformer_blocks = 1
    pipeline.frame_seq_length = 1
    pipeline.kv_cache_size = 8
    pipeline.generate_and_sync_list = lambda blocks, steps, device: [selected] * blocks
    pipeline._initialize_crossattn_cache = types.MethodType(initialize_small_cross_cache, pipeline)
    if baseline:
        pipeline._initialize_kv_cache = types.MethodType(initialize_small_baseline_cache, pipeline)
    return pipeline, student


NOISE = torch.tensor([0.25, -0.4], dtype=torch.float64).reshape(1, 2, 1, 1, 1)


def run_rollout(selected=1, alpha=1.0, history_grad=True, grad=True, baseline=False):
    pipeline, student = make_pipeline(selected, alpha, history_grad, baseline)
    torch.manual_seed(1234)
    with torch.set_grad_enabled(grad):
        output, timestep_from, timestep_to = pipeline.inference_with_trajectory(noise=NOISE.clone())
    return output, student, pipeline, (timestep_from, timestep_to)


class STEPipelineTests(unittest.TestCase):
    def test_all_four_steps_run_but_only_selected_and_writers_record_graph(self):
        output, student, _, _ = run_rollout(selected=1)
        self.assertEqual([call["timestep"] for call in student.calls], [900, 700, 400, 100, 0] * 2)
        self.assertEqual(
            [call["grad_enabled"] for call in student.calls],
            [False, True, False, False, True] * 2,
        )
        self.assertTrue(output.requires_grad)
        for call in student.calls:
            self.assertEqual(call["output"].requires_grad, call["grad_enabled"])

    def test_history_is_full_four_step_result_independent_of_selection_and_alpha(self):
        _, reference_student, _, _ = run_rollout(selected=3, alpha=1.0, grad=False)
        for selected in range(4):
            for alpha in (0.0, 0.25, 1.0, 2.0):
                with self.subTest(selected=selected, alpha=alpha):
                    output, student, _, _ = run_rollout(selected=selected, alpha=alpha)
                    for frame in range(2):
                        writer = student.calls[frame * 5 + 4]
                        terminal = student.calls[frame * 5 + 3]
                        reference = reference_student.calls[frame * 5 + 4]
                        torch.testing.assert_close(writer["input"].detach(), reference["input"], rtol=0, atol=0)
                        torch.testing.assert_close(writer["input"].detach(), terminal["output"].detach(), rtol=0, atol=0)
                        # DMD continues to supervise selected-step samples.
                        selected_call = student.calls[frame * 5 + selected]
                        torch.testing.assert_close(output[:, frame:frame + 1].detach(), selected_call["output"].detach(), rtol=0, atol=0)

    def test_alpha_scales_history_gradient_but_not_direct_selected_loss(self):
        for alpha in (0.0, 0.25, 1.0, 2.0):
            with self.subTest(alpha=alpha):
                output, student, _, _ = run_rollout(selected=1, alpha=alpha)
                output[:, :1].sum().backward()
                torch.testing.assert_close(
                    student.calls[1]["output"].grad,
                    torch.ones_like(student.calls[1]["output"]),
                )
                self.assertFalse(student.calls[1]["input"].requires_grad)
                self.assertFalse(student.calls[2]["input"].requires_grad)
                self.assertFalse(student.calls[3]["input"].requires_grad)
                future_output, future_student, _, _ = run_rollout(selected=1, alpha=alpha)
                future_output[:, 1:].sum().backward()
                torch.testing.assert_close(
                    future_student.calls[1]["output"].grad,
                    torch.full_like(future_student.calls[1]["output"], 0.2 * 0.7 * alpha),
                )
                self.assertIsNotNone(future_student.writer_weight.grad)
                self.assertGreater(future_student.writer_weight.grad.abs().item(), 0)

    def test_future_only_loss_reaches_earlier_selected_prediction_through_history(self):
        output, student, _, _ = run_rollout(selected=1)
        output[:, 1:].sum().backward()
        earlier_gradient = student.calls[1]["output"].grad
        self.assertIsNotNone(earlier_gradient)
        self.assertGreater(earlier_gradient.abs().sum().item(), 0)
        writer_input_gradient = student.calls[4]["input"].grad
        self.assertIsNotNone(writer_input_gradient)
        self.assertGreater(writer_input_gradient.abs().sum().item(), 0)
        # Disabling history gradients keeps the same samples while cutting the
        # only path from the later-frame loss to the earlier prediction.
        detached, detached_student, _, _ = run_rollout(selected=1, history_grad=False)
        torch.testing.assert_close(detached.detach(), output.detach(), rtol=0, atol=0)
        detached[:, 1:].sum().backward()
        earlier_gradient = detached_student.calls[1]["output"].grad
        self.assertTrue(earlier_gradient is None or torch.count_nonzero(earlier_gradient) == 0)

    def test_no_grad_critic_sampling_uses_identical_full_rollout(self):
        trained, _, _, bounds = run_rollout(selected=1, alpha=0.25)
        sampled, student, _, sampled_bounds = run_rollout(selected=1, alpha=0.25, grad=False)
        torch.testing.assert_close(sampled, trained.detach(), rtol=0, atol=0)
        self.assertEqual(bounds, sampled_bounds)
        self.assertFalse(sampled.requires_grad)
        self.assertTrue(all(not call["grad_enabled"] for call in student.calls))
        self.assertEqual(len(student.calls), 10)

    def test_original_pipeline_still_truncates_and_detaches_writer(self):
        output, student, _, bounds = run_rollout(selected=1, baseline=True)
        self.assertEqual([call["timestep"] for call in student.calls], [900, 700, 0] * 2)
        self.assertEqual([call["grad_enabled"] for call in student.calls], [False, True, False] * 2)
        self.assertEqual(bounds, (None, None))
        for frame, selected_call in enumerate((student.calls[1], student.calls[4])):
            torch.testing.assert_close(output[:, frame:frame + 1].detach(), selected_call["output"].detach())
        output[:, 1:].sum().backward()
        earlier_gradient = student.calls[1]["output"].grad
        self.assertTrue(earlier_gradient is None or torch.count_nonzero(earlier_gradient) == 0)

    def test_dmd_critic_receives_selected_samples_and_keeps_student_frozen(self):
        expected, _, _, _ = run_rollout(selected=1)
        pipeline, student = make_pipeline(selected=1)
        dmd = DMD.__new__(DMD)
        nn.Module.__init__(dmd)
        dmd.scheduler = pipeline.scheduler
        dmd.args = types.SimpleNamespace(denoising_loss_type="noise")
        dmd.ts_schedule = False
        dmd.ts_schedule_max = False
        dmd.min_score_timestep = 0
        dmd.num_train_timestep = 1000
        dmd.num_frame_per_block = 1
        dmd.timestep_shift = 1.0
        dmd.min_step = 0
        dmd.max_step = 1000
        dmd._get_timestep = lambda *args, **kwargs: torch.zeros(1, 2, dtype=torch.long)
        dmd.fake_score = lambda noisy_image_or_video, **kwargs: (None, noisy_image_or_video * 0.9)
        observed = {}

        def run_generator(**kwargs):
            torch.manual_seed(1234)
            sample, begin, end = pipeline.inference_with_trajectory(noise=NOISE.clone())
            return sample, None, begin, end

        def critic_objective(**kwargs):
            observed.update(kwargs)
            return kwargs["x_pred"].square().mean()

        dmd._run_generator = run_generator
        dmd.denoising_loss_func = critic_objective
        dmd.critic_loss(list(NOISE.shape), {}, {}, clean_latent=None)
        torch.testing.assert_close(observed["x"], expected.detach().flatten(0, 1), rtol=0, atol=0)
        self.assertFalse(observed["x"].requires_grad)
        self.assertTrue(all(not call["grad_enabled"] for call in student.calls))
        self.assertEqual(len(student.calls), 10)


class CountingSink(dict):
    def __init__(self):
        super().__init__()
        self.updates = 0

    def update(self, *args, **kwargs):
        self.updates += 1
        return super().update(*args, **kwargs)


class TinyAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden, kv_cache, current_start, crossattn_cache, ste_return_kv=False):
        self.calls += 1
        if crossattn_cache is not None:
            raise AssertionError("STE must recompute cross-attention projections")
        query = (hidden * 0.7).unsqueeze(2)
        key = hidden.sin().unsqueeze(2)
        value = hidden.square().unsqueeze(2)
        result = cached_attention(query, key, value, kv_cache, current_start, 32, -1)
        hidden = result.squeeze(2).sin()
        if ste_return_kv:
            return hidden, key, value
        return hidden


def make_small_wan_wrapper(checkpointing):
    """Construct the real model/wrapper with tiny CPU layers and no weight load."""
    model = CausalModel.__new__(CausalModel)
    nn.Module.__init__(model)
    model.model_type = "t2v"
    model.dim = 4
    model.freq_dim = 4
    model.text_len = 2
    model.out_dim = 1
    model.patch_size = (1, 1, 1)
    model.gradient_checkpointing = checkpointing
    model.block_mask = None
    model.freqs = torch.ones(16, 2, dtype=torch.complex128)
    model.patch_embedding = nn.Conv3d(1, 4, kernel_size=1)
    model.time_embedding = nn.Linear(4, 4)
    model.time_projection = nn.Linear(4, 24)
    model.text_embedding = nn.Linear(4, 4)
    model.blocks = nn.ModuleList([
        CausalBlock("test", dim=4, ffn_dim=8, num_heads=1, qk_norm=False)
    ])
    model.head = CausalHead(dim=4, out_dim=1, patch_size=(1, 1, 1))
    model.double()
    wrapper = WanWrapper.__new__(WanWrapper)
    nn.Module.__init__(wrapper)
    wrapper.model = model
    wrapper.uniform_timestep = False
    wrapper.seq_len = 8
    wrapper.scheduler = ToyScheduler()
    return wrapper


class STECacheTests(unittest.TestCase):
    def test_real_block_model_and_wrapper_return_writer_graph_publicly(self):
        for checkpointing in (False, True):
            with self.subTest(checkpointing=checkpointing):
                torch.manual_seed(71)
                wrapper = make_small_wan_wrapper(checkpointing)
                writer_input = torch.tensor([0.35], dtype=torch.float64).reshape(1, 1, 1, 1, 1).requires_grad_()
                conditional = {"prompt_embeds": torch.tensor([[[0.2, -0.3, 0.4, 0.1]]], dtype=torch.float64)}
                writer_cache = [{
                    "ste_cache": True, "ste_k_chunks": (), "ste_v_chunks": (),
                    "ste_cache_output": {},
                }]
                boundary_outputs = {}
                handles = [
                    wrapper.model.blocks[0].register_forward_hook(
                        lambda module, args, output: boundary_outputs.update(block=output)
                    ),
                    wrapper.model.register_forward_hook(
                        lambda module, args, output: boundary_outputs.update(model=output)
                    ),
                ]
                writer_result = wrapper(
                    writer_input, conditional, torch.zeros(1, 1, dtype=torch.long),
                    kv_cache=writer_cache, crossattn_cache=[None], current_start=0,
                )
                for handle in handles:
                    handle.remove()
                self.assertEqual(len(writer_result), 3)
                self.assertIsInstance(boundary_outputs["block"], tuple)
                self.assertEqual(len(boundary_outputs["block"]), 3)
                self.assertIsInstance(boundary_outputs["model"], tuple)
                self.assertEqual(len(boundary_outputs["model"]), 2)
                key, value = writer_result[2][0]
                self.assertTrue(key.requires_grad)
                self.assertTrue(value.requires_grad)
                self.assertIs(boundary_outputs["block"][1], key)
                self.assertIs(boundary_outputs["block"][2], value)
                self.assertIs(boundary_outputs["model"][1][0][0], key)
                self.assertIs(boundary_outputs["model"][1][0][1], value)
                key.retain_grad()
                value.retain_grad()
                reader_cache = [{
                    "ste_cache": True, "ste_k_chunks": (key,), "ste_v_chunks": (value,),
                }]
                # Subsequent callers consume only the public return; the
                # private writer dictionary is disposable after the call.
                writer_cache[0]["ste_cache_output"].clear()
                reader_result = wrapper(
                    torch.full_like(writer_input, -0.2), conditional,
                    torch.full((1, 1), 700, dtype=torch.long),
                    kv_cache=reader_cache, crossattn_cache=[None], current_start=1,
                )
                self.assertEqual(len(reader_result), 2)
                reader_result[1].square().sum().backward()
                self.assertGreater(writer_input.grad.abs().sum().item(), 0)
                self.assertGreater(value.grad.abs().sum().item(), 0)
                self.assertIsNotNone(key.grad)
                context_gradient = wrapper.model.blocks[0].cross_attn.projection.weight.grad
                self.assertGreater(context_gradient.abs().sum().item(), 0)

    def test_immutable_cache_matches_original_attention_across_writes_and_reads(self):
        torch.manual_seed(99)
        layer = CausalAttention(dim=4, num_heads=1, qk_norm=False).double()
        baseline = {
            "k": torch.zeros(1, 4, 1, 4, dtype=torch.float64),
            "v": torch.zeros(1, 4, 1, 4, dtype=torch.float64),
            "global_end_index": torch.tensor([0]),
            "local_end_index": torch.tensor([0]),
        }
        history = {"ste_cache": True, "ste_k_chunks": (), "ste_v_chunks": ()}
        kwargs = {
            "seq_lens": torch.tensor([1]),
            "grid_sizes": torch.tensor([[1, 1, 1]]),
            "freqs": torch.ones(16, 2, dtype=torch.complex128),
            "block_mask": None,
        }
        with torch.no_grad():
            for start in range(2):
                latent = torch.randn(1, 1, 4, dtype=torch.float64)
                sink = {}
                view = dict(history, ste_cache_output=sink)
                expected = layer(latent, kv_cache=baseline, current_start=start, **kwargs)
                actual = layer(latent, kv_cache=view, current_start=start, **kwargs)
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                history["ste_k_chunks"] += (sink["k"],)
                history["ste_v_chunks"] += (sink["v"],)

            committed_keys = torch.cat(history["ste_k_chunks"], dim=1).clone()
            committed_values = torch.cat(history["ste_v_chunks"], dim=1).clone()
            for _ in range(3):
                latent = torch.randn(1, 1, 4, dtype=torch.float64)
                expected = layer(latent, kv_cache=baseline, current_start=2, **kwargs)
                actual = layer(latent, kv_cache=dict(history), current_start=2, **kwargs)
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(torch.cat(history["ste_k_chunks"], dim=1), committed_keys, rtol=0, atol=0)
                torch.testing.assert_close(torch.cat(history["ste_v_chunks"], dim=1), committed_values, rtol=0, atol=0)
        self.assertEqual(len(history["ste_k_chunks"]), 2)

    def test_checkpoint_replay_preserves_history_gradient_and_commits_only_once(self):
        def simulate(checkpointing):
            block = TinyAttentionBlock()
            first = torch.tensor([[[0.4, -0.2]]], dtype=torch.float64, requires_grad=True)
            second = torch.tensor([[[0.1, 0.5]]], dtype=torch.float64, requires_grad=True)
            sink = CountingSink()
            writer_view = {
                "ste_cache": True, "ste_k_chunks": (), "ste_v_chunks": (),
                "ste_cache_output": sink,
            }
            writer_kwargs = {
                "kv_cache": writer_view, "current_start": 0,
                "crossattn_cache": {"is_init": True},
            }
            forward_ste_block(block, first, writer_kwargs, checkpointing)
            key, value = sink["k"], sink["v"]
            reader_view = {
                "ste_cache": True, "ste_k_chunks": (key,), "ste_v_chunks": (value,),
            }
            result = forward_ste_block(
                block, second,
                {"kv_cache": reader_view, "current_start": 1, "crossattn_cache": {}},
                checkpointing,
            )
            # The live rollout advances while checkpoint backward still needs
            # the exact prefix used by each original call.
            writer_view["ste_k_chunks"] = (key,)
            writer_view["ste_v_chunks"] = (value,)
            reader_view["ste_k_chunks"] = ()
            reader_view["ste_v_chunks"] = ()
            result.square().sum().backward()
            self.assertEqual(sink.updates, 1)
            self.assertIs(sink["k"], key)
            self.assertIs(sink["v"], value)
            self.assertGreater(first.grad.abs().sum().item(), 0)
            if checkpointing:
                self.assertGreater(block.calls, 2)
            return result.detach(), first.grad, second.grad

        expected = simulate(False)
        checkpointed = simulate(True)
        for actual, reference in zip(checkpointed, expected):
            torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)

    def test_invalid_history_position_and_rolling_window_fail_explicitly(self):
        token = torch.ones(1, 1, 1, 2)
        cache = {"ste_cache": True, "ste_k_chunks": (), "ste_v_chunks": ()}
        with self.assertRaisesRegex(ValueError, "current_start"):
            cached_attention(token, token, token, cache, 1, 32, -1)
        with self.assertRaisesRegex(ValueError, "local_attn_size"):
            cached_attention(token, token, token, cache, 0, 32, 4)


if __name__ == "__main__":
    unittest.main()
