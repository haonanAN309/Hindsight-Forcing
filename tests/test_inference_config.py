import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('release_inference',ROOT/'inference.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
from release_runtime.inference_worker import selected_state, source_for_task, inference_policy_suites

class InferenceConfigTests(unittest.TestCase):
    def args(self,method,suite='short',**kwargs):
        return SimpleNamespace(method=method,suite=suite,weights=kwargs.get('weights','evaluation'),checkpoint=None,weight_key=None,task_file=None,prompt='A person walking.',prompt_file=None,seed=42)
    def test_all_adapters_both_lengths(self):
        for method in module.METHODS:
            for suite in ('short','long'):
                with self.subTest(method=method,suite=suite):
                    manifest,tasks=module.make_plan(self.args(method,suite))
                    m=manifest['models'][0]
                    self.assertTrue(Path(m['config_path']).is_file())
                    self.assertTrue(Path(source_for_task(m,tasks[0])).is_dir())
                    self.assertEqual(tasks[0]['seed'],42)
    def test_paper_hf_policy(self):
        manifest,_=module.make_plan(self.args('hf'))
        m=manifest['models'][0]
        self.assertEqual(m['weight_key'],'generator_ema')
        self.assertTrue(m['source_root'].endswith('/engines/hf'))
        self.assertEqual(m['long_policy']['cache_latents'],12)
        self.assertEqual(m['long_policy']['position_mode'],'top_aligned')
    def test_raw_weight_selection(self):
        for method in ('hf','hf_long','raven','reward_forcing','sgf','sgf_plus'):
            manifest,_=module.make_plan(self.args(method,weights='ordinary'))
            self.assertEqual(manifest['models'][0]['weight_key'],'generator')
    def test_hf_long_matches_evaluated_cache_and_checkpoint(self):
        for suite in ('short', 'long'):
            manifest, tasks = module.make_plan(self.args('hf_long', suite))
            m = manifest['models'][0]
            self.assertTrue(m['checkpoint_path'].endswith('/checkpoints/hf_long.pt'))
            self.assertEqual(m['weight_key'], 'generator_ema')
            self.assertTrue(m['config_path'].endswith('/configs/hf_long_inference.yaml'))
            self.assertTrue(source_for_task(m, tasks[0]).endswith('/engines/sgf'))
            self.assertIn(suite, inference_policy_suites(m))
            self.assertEqual(m['long_policy'], dict(kind='relative_rope_fifo', cache_latents=12,
                sink_latents=3, train_latents=12, position_mode='contiguous'))
            self.assertEqual(m['config_overrides']['model_kwargs'], dict(local_attn_size=9, sink_size=3))
    def test_official_weights_not_silently_changed(self):
        with self.assertRaises(ValueError):module.make_plan(self.args('self_forcing',weights='ordinary'))
    def test_state_prefixes_and_no_fallback(self):
        token=object()
        self.assertEqual(selected_state({'generator_ema':{'model._fsdp_wrapped_module._orig_mod.x':token}},'generator_ema'),{'model.x':token})
        with self.assertRaises(ValueError):selected_state({'generator':{'x':token}},'generator_ema')
        with self.assertRaises(ValueError):selected_state({'generator':{'model.x':token,'model._orig_mod.x':token}},'generator')

if __name__=='__main__': unittest.main()
