#!/usr/bin/env python3
"""Generate short or one-minute videos with the evaluated model adapters."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from release_runtime.inference_worker import atomic_json, read_tasks, run, validate_plan

ROOT = Path(__file__).resolve().parent
METHODS = ('hf', 'hf_long', 'causvid', 'self_forcing', 'causal_forcing', 'reward_forcing', 'raven', 'sgf', 'sgf_plus')


def make_plan(args):
    manifest = json.loads((ROOT / 'configs/inference.json').read_text())
    manifest['asset_root'] = str(ROOT)
    model = next(m for m in manifest['models'] if m['id'] == args.method)
    manifest['models'] = [model]
    for field in ('source_root', 'long_source_root', 'config_path', 'checkpoint_path'):
        if field in model:
            model[field] = str((ROOT / model[field]).resolve())
    if args.weights == 'ordinary':
        index = json.loads((ROOT / 'checkpoints/model_index.json').read_text())
        entry = next(m for m in index['checkpoints'] if m['id'] == args.method)
        ordinary = next((f for f in entry['files'] if f['role'] == 'ordinary'), None)
        if ordinary is None:
            raise ValueError('This official checkpoint has no separate ordinary-weight entry; use --weights evaluation.')
        model['checkpoint_path'] = str((ROOT / ordinary['path']).resolve())
        model['weight_key'] = ordinary['state_dict_key']
    if args.checkpoint:
        model['checkpoint_path'] = str(Path(args.checkpoint).expanduser().resolve())
    if args.weight_key:
        model['weight_key'] = args.weight_key
    if args.task_file:
        tasks = read_tasks(args.task_file)
        if any(task.get('suite') != args.suite for task in tasks):
            raise ValueError('Each task suite must match --suite; run short and long tasks in separate processes.')
    else:
        prompts = [args.prompt] if args.prompt else [line.strip() for line in Path(args.prompt_file).read_text().splitlines() if line.strip()]
        if not prompts:
            raise ValueError('No prompts supplied')
        tasks = [{'id': f'sample_{i:04d}', 'prompt': p, 'seed': (args.seed+i) % 2**32, 'suite': args.suite} for i,p in enumerate(prompts)]
    validate_plan(manifest, args.method, tasks)
    return manifest, tasks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, default='hf')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prompt')
    group.add_argument('--prompt-file', help='One expanded generation prompt per line')
    group.add_argument('--task-file', help='JSON/JSONL tasks with id, prompt, seed, suite')
    parser.add_argument('--suite', choices=('short','long'), default='short')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='outputs/inference')
    parser.add_argument('--weights', choices=('evaluation','ordinary'), default='evaluation')
    parser.add_argument('--checkpoint', help='Optional checkpoint path override')
    parser.add_argument('--weight-key', choices=('generator','generator_ema'), help='Explicit state-dict key override')
    parser.add_argument('--dry-run', action='store_true', help='Validate the plan without loading weights or CUDA')
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error('--seed must be in [0, 2**32)')
    try:
        manifest, tasks = make_plan(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps({'model':manifest['models'][0], 'video':manifest['video'], 'tasks':tasks}, indent=2))
        return
    model=manifest['models'][0]
    for field in ('source_root','config_path','checkpoint_path'):
        if not Path(model[field]).exists():
            parser.error(f'Missing {field}: {model[field]}. See checkpoints/README.md and docs/DEPENDENCIES.md.')
    output=Path(args.output_dir).expanduser().resolve()
    plan_dir=output/'plans'/f'{args.method}_{args.suite}'
    manifest_path=plan_dir/'manifest.json'
    task_path=plan_dir/'tasks.json'
    atomic_json(manifest_path,manifest)
    atomic_json(task_path,tasks)
    run(SimpleNamespace(manifest=str(manifest_path), model_id=args.method, task_file=str(task_path), output_root=str(output), validate_only=False))


if __name__ == '__main__':
    main()
