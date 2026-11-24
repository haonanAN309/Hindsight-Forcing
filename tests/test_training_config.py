"""CPU configuration, budget and native-entry contracts for the public launcher."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from release_runtime.training_config import (
    METHODS, accumulation_for_batch, build_training_config, engine_files,
    training_plan, validate_distributed_environment, validate_training_assets,
    validate_training_output,
)


class TrainingConfigurationTests(unittest.TestCase):
    def test_optimizer_batch_is_preserved_across_deployments(self):
        for method in METHODS:
            for world in (1, 2, 4):
                for batch in (1, 2):
                    with self.subTest(method=method, world=world, batch=batch):
                        cfg = build_training_config(method, env={"WORLD_SIZE": str(world)}, micro_batch=batch)
                        self.assertEqual(world * batch * cfg.gradient_accumulation_steps, 64)
                        self.assertEqual(cfg.image_or_video_shape[0], batch)

    def test_invalid_batch_and_accumulation_fail(self):
        for args in ((64, 3, 1), (64, 1, 0), (64, 0, 1), (64, 1, 1, 2), (64, True, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                accumulation_for_batch(*args)

    def test_native_budgets_ema_and_flags(self):
        for method in METHODS:
            with self.subTest(method=method):
                cfg = build_training_config(method, env={})
                self.assertEqual(cfg.max_steps, 300 if method == "raven" else 750)
                self.assertEqual(training_plan(method, cfg, env={})["student_updates"], 150)
                self.assertEqual(cfg.ema_weight, 0.99)
                self.assertEqual(cfg.seed, 20261020)
                self.assertTrue(cfg.save_student_and_ema)
                self.assertFalse(cfg.no_save)
                self.assertTrue(cfg.no_visualize)
                self.assertTrue(cfg.disable_wandb)
                self.assertEqual(cfg.context_noise, 0)
                self.assertEqual(cfg.num_training_frames, 21)
                if method != "hf":
                    self.assertEqual(cfg.ema_start_student_update, 40)
                else:
                    self.assertEqual(cfg.ema_start_step, 200)

    def test_smoke_budget_preserves_the_real_computation(self):
        for method, (_, _, ratio) in METHODS.items():
            with self.subTest(method=method):
                full = build_training_config(method, env={"WORLD_SIZE": "2"})
                smoke = build_training_config(method, smoke_test=True, env={"WORLD_SIZE": "2"})
                plan = training_plan(method, smoke, env={"WORLD_SIZE": "2"})
                self.assertTrue(plan["smoke_test"])
                self.assertEqual((plan["student_updates"], plan["critic_updates"], plan["global_steps"]), (1, ratio, ratio))
                self.assertEqual(plan["micro_batch"] * plan["world_size"] * plan["accumulation_steps"], 64)
                self.assertEqual(smoke.logdir, f"outputs/{method}_smoke")
                allowed = {"release_smoke_test", "logdir", "max_steps"}
                if method == "hf":
                    allowed.add("ema_start_step")
                    self.assertEqual(smoke.ema_start_step, 0)
                else:
                    allowed.update(("target_student_updates", "ema_start_student_update"))
                    self.assertEqual(smoke.ema_start_student_update, 0)
                full_values = OmegaConf.to_container(full, resolve=True)
                smoke_values = OmegaConf.to_container(smoke, resolve=True)
                changed = {key for key in set(full_values) | set(smoke_values)
                           if full_values.get(key) != smoke_values.get(key)}
                self.assertEqual(changed, allowed)

    def test_smoke_still_validates_the_original_training_budget(self):
        loader = OmegaConf.load
        def malformed_config(path):
            config = loader(path)
            if "max_steps" in config:
                config.max_steps = 1
            return config
        with mock.patch("release_runtime.training_config.OmegaConf.load", side_effect=malformed_config):
            with self.assertRaisesRegex(ValueError, "150 Student updates"):
                build_training_config("hf", smoke_test=True, env={})

    def test_smoke_output_is_separate_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_output = root / "training"
            full_output.mkdir()
            (full_output / "checkpoint.pt").write_bytes(b"keep")
            config = build_training_config("hf", smoke_test=True, output_dir=str(full_output), env={})
            self.assertEqual(config.logdir, str(root / "training_smoke"))
            validate_training_output(config)
            self.assertFalse(Path(config.logdir).exists())
            Path(config.logdir).mkdir()
            validate_training_output(config)
            (Path(config.logdir) / "config.yaml").write_text("existing run")
            with self.assertRaisesRegex(ValueError, "not empty"):
                validate_training_output(config)
            self.assertEqual((full_output / "checkpoint.pt").read_bytes(), b"keep")
            symlink = root / "linked_smoke"
            symlink.symlink_to(full_output, target_is_directory=True)
            linked = build_training_config("hf", smoke_test=True, output_dir=str(symlink), env={})
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                validate_training_output(linked)
            with self.assertRaises(ValueError):
                build_training_config("hf", smoke_test=True, output_dir="/", env={})

    def test_hf_smoke_ema_is_rebuilt_from_loaded_student_only(self):
        spec = importlib.util.spec_from_file_location("release_train", ROOT / "train.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded_student = object()
        for method in METHODS:
            for smoke_test in (False, True):
                with self.subTest(method=method, smoke_test=smoke_test):
                    previous_ema = object()
                    new_ema = object()
                    trainer = SimpleNamespace(model=SimpleNamespace(generator=loaded_student), generator_ema=previous_ema)
                    factory = mock.Mock(return_value=new_ema)
                    config = build_training_config(method, smoke_test=smoke_test, env={})
                    module.initialize_smoke_ema(SimpleNamespace(EMA_FSDP=factory), trainer, method, config)
                    if smoke_test and method == "hf":
                        factory.assert_called_once_with(loaded_student, decay=0.99)
                        self.assertIs(trainer.generator_ema, new_ema)
                    else:
                        factory.assert_not_called()
                        self.assertIs(trainer.generator_ema, previous_ema)

    def test_hf_rejects_unsupported_training_options(self):
        loader = OmegaConf.load

        def unsupported_config(path):
            config = loader(path)
            config.ste_unknown_option = True
            return config

        with mock.patch("release_runtime.training_config.OmegaConf.load", side_effect=unsupported_config):
            with self.assertRaisesRegex(ValueError, "Unsupported option"):
                build_training_config("hf", env={})

    def test_method_specific_computations_are_preserved(self):
        hf = build_training_config("hf", env={})
        self.assertTrue(hf.ste_training and hf.ste_history_grad)
        self.assertEqual(hf.ste_alpha, 1)
        self.assertTrue(hf.same_step_across_blocks)
        with self.assertRaisesRegex(ValueError, "Unknown method"):
            build_training_config("hf_long", env={})
        raven = build_training_config("raven", env={})
        self.assertEqual(list(raven.denoising_step_list), [1000, 938, 833, 625])
        self.assertFalse(raven.warp_denoising_step)
        self.assertEqual(raven.raven_chunk_weighting, -1)
        self.assertTrue(raven.raven_rollout_reuse and raven.raven_critic_first)
        rf = build_training_config("reward_forcing", env={})
        self.assertEqual(rf.reward_inverse_temperature, 2)
        self.assertEqual(rf.reward_sink_decay, 0.999)
        self.assertEqual(rf.reward_training_cache_frames, 21)
        self.assertEqual(rf.reward_inference_cache_frames, 9)
        for method in ("sgf", "sgf_plus"):
            cfg = build_training_config(method, env={})
            self.assertTrue(cfg.per_rank_exit_step)
            self.assertTrue(cfg.self_gradient_forcing_match_context)
            self.assertEqual(cfg.self_gradient_forcing_cache_mode, "exit")
            self.assertEqual(bool(cfg.model_kwargs.get("dual_full_expert", {}).get("enabled", False)), method == "sgf_plus")

    def test_sgf_native_config_validators_accept_public_configs(self):
        for method in ("sgf", "sgf_plus"):
            engine, _, _, _ = engine_files(method)
            tree = ast.parse((engine / "trainer/matched_runtime.py").read_text())
            selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in ("training_budget", "validate_config")]
            namespace = {"EXPECTED_METHOD": method}
            exec(compile(ast.Module(body=selected, type_ignores=[]), str(engine), "exec"), namespace)
            cfg = build_training_config(method, env={"WORLD_SIZE": "2"})
            self.assertEqual(namespace["validate_config"](cfg, 2), (150, 5, 750))
            smoke = build_training_config(method, smoke_test=True, env={"WORLD_SIZE": "2"})
            self.assertEqual(namespace["validate_config"](smoke, 2), (1, 5, 5))

    def test_user_paths_and_seed_override(self):
        cfg = build_training_config("hf", data_path="data/prompts.txt", init_checkpoint="weights/initial.pt", output_dir="outputs/example", seed=37, env={})
        self.assertEqual((cfg.data_path, cfg.generator_ckpt, cfg.logdir, cfg.seed), ("data/prompts.txt", "weights/initial.pt", "outputs/example", 37))
        default = build_training_config("hf", env={})
        self.assertEqual(default.generator_ckpt, "checkpoints/initialization/causal_ode.pt")
        with self.assertRaises(ValueError):
            build_training_config("hf", reward_model_path="weights/reward", env={})

    def test_asset_checks_do_not_load_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prompts.txt").write_text("A test prompt\n")
            (root / "initial.pt").write_bytes(b"existence-check-only")
            cfg = build_training_config("hf", data_path="prompts.txt", init_checkpoint="initial.pt", env={})
            validate_training_assets("hf", cfg, root)
            (root / "initial.pt").unlink()
            with self.assertRaises(FileNotFoundError):
                validate_training_assets("hf", cfg, root)

    def test_torchrun_environment_is_required_for_real_training(self):
        with self.assertRaises(ValueError):
            validate_distributed_environment({})
        validate_distributed_environment({"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "localhost", "MASTER_PORT": "29500"})
        with self.assertRaises(ValueError):
            validate_distributed_environment({"RANK": "1", "LOCAL_RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "localhost", "MASTER_PORT": "29500"})

    def test_dry_run_never_imports_torch_or_creates_outputs(self):
        script = """
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('release_train', Path(sys.argv[1]) / 'train.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
module.main(['--method', sys.argv[2], '--dry-run', '--output-dir', sys.argv[3]])
assert 'torch' not in sys.modules
assert not Path(sys.argv[3]).exists()
"""
        with tempfile.TemporaryDirectory() as directory:
            for method in METHODS:
                destination = str(Path(directory) / method)
                environment = dict(os.environ, WORLD_SIZE="2", CUDA_VISIBLE_DEVICES="")
                result = subprocess.run([sys.executable, "-c", script, str(ROOT), method, destination], cwd=ROOT, env=environment, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                data = json.loads(result.stdout)
                self.assertEqual(data["student_updates"], 150)
                self.assertEqual(data["micro_batch"] * data["world_size"] * data["accumulation_steps"], 64)

    def test_smoke_cli_dry_run_reports_actual_budget_without_cuda(self):
        script = """
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('release_train', Path(sys.argv[1]) / 'train.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.main(['--method', sys.argv[2], '--smoke-test', '--dry-run', '--output-dir', sys.argv[3]])
assert 'torch' not in sys.modules
assert not Path(sys.argv[3] + '_smoke').exists()
"""
        with tempfile.TemporaryDirectory() as directory:
            for method, (_, _, ratio) in METHODS.items():
                with self.subTest(method=method):
                    destination = str(Path(directory) / method)
                    environment = dict(os.environ, WORLD_SIZE="2", CUDA_VISIBLE_DEVICES="")
                    result = subprocess.run([sys.executable, "-c", script, str(ROOT), method, destination], cwd=ROOT, env=environment, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    plan = json.loads(result.stdout)
                    self.assertTrue(plan["smoke_test"])
                    self.assertEqual((plan["student_updates"], plan["critic_updates"]), (1, ratio))
                    self.assertEqual(plan["config"]["logdir"], destination + "_smoke")

    def test_native_trainers_import_without_initializing_cuda(self):
        script = """
import importlib, sys
from pathlib import Path
root, engine, module = sys.argv[1:]
sys.path.insert(0, str(Path(root) / 'engines' / engine))
import torch
assert not torch.cuda.is_initialized()
trainer = importlib.import_module(module).Trainer
assert callable(trainer)
assert not torch.cuda.is_initialized()
"""
        with tempfile.TemporaryDirectory() as cache:
            for method, (engine, module, _) in METHODS.items():
                with self.subTest(method=method):
                    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HOME=cache)
                    result = subprocess.run([sys.executable, "-c", script, str(ROOT), engine, module], cwd=ROOT, env=environment, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
