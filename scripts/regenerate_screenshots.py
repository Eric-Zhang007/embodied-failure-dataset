"""
Replay ALFRED trajectories to regenerate missing screenshots.
Does NOT overwrite eb_reasoning — reads existing JSON, replays to get screenshots, restores reasoning.
Supports parallel workers.

Usage:
    uv run python scripts/regenerate_screenshots.py --workers 8
"""
import argparse, glob, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.env_controller import EnvController
from src.alfred_parser import load_traj, extract_metadata, extract_low_actions
from src.action_adapter import adapt, resolve_object_ids
from src.step_recorder import StepRecorder

def expert_params_to_types(action, params):
    p = dict(params)
    for k, v in list(p.items()):
        if isinstance(v, str) and '|' in v:
            tn = v.split('|')[0]
            if 'receptacle' in k.lower():
                p['receptacleType'] = tn; p.pop(k, None)
            elif k.lower() == 'objectid':
                p['objectType'] = tn
    p.pop('forceAction', None); p.pop('moveMagnitude', None)
    return p

def process_one(args):
    ep_path, data_dir, output_dir = args
    eid = os.path.basename(ep_path).replace('.json', '')
    img_dir = os.path.join(output_dir, eid)
    
    # Already has screenshots? Skip
    if os.path.isdir(img_dir):
        return 'skip', eid
    
    # Load existing reasoning
    with open(ep_path) as f:
        old_ep = json.load(f)
    old_steps = old_ep.get('steps', [])
    old_reasoning = {s['step_id']: s.get('eb_reasoning') for s in old_steps}
    
    task_type = old_ep.get('task_type', '')
    scene = old_ep.get('scene', '')
    if not scene:
        raise ValueError(f'{eid}: episode JSON lacks scene')
    
    # Find the original ALFRED traj_data.json
    task_id = old_ep.get('alfred_task_id', '')
    # Search for it in data/json_2.1.0
    traj_path = None
    for root, dirs, files in os.walk(data_dir):
        if task_id in dirs:
            traj_path = os.path.join(root, task_id, 'traj_data.json')
            break
    if not traj_path:
        raise FileNotFoundError(f'{eid}: original ALFRED traj_data.json not found')
    
    traj = load_traj(traj_path)
    meta = extract_metadata(traj)
    env = EnvController(scene=meta['scene'])
    recorder = StepRecorder()
    
    try:
        env.reset_to_alfred_scene(meta['alfred_scene'])

        low = extract_low_actions(traj)
        for i, la in enumerate(low):
            typed = expert_params_to_types(la['action'], la['params'])
            state = env.get_state_snapshot()
            resolved, warning = resolve_object_ids(la['action'], typed, state['metadata'].get('objects', []))
            if warning:
                raise ValueError(f'{eid} step {i}: {warning}')
            exec_act, exec_params = adapt(la['action'], resolved)

            sid = f's{i}'
            screenshot_path = os.path.join(img_dir, f'{sid}.png')
            os.makedirs(img_dir, exist_ok=True)
            recorder.save_frame(state['frame'], screenshot_path)

            result = env.step(exec_act, **exec_params)
            if not result['success']:
                raise RuntimeError(f'{eid} step {i}: {exec_act} failed: {result["error"]}')

        # Restore reasoning into JSON
        for s in old_steps:
            sid = s.get('step_id', '')
            s['image_path'] = os.path.join(output_dir, eid, f'{sid}.png')
            if s.get('eb_reasoning'):
                pass  # keep existing
            elif sid in old_reasoning and old_reasoning[sid]:
                s['eb_reasoning'] = old_reasoning[sid]
            else:
                s['eb_reasoning'] = None
        
        old_ep['steps'] = old_steps
        with open(ep_path, 'w') as f:
            json.dump(old_ep, f, indent=2, ensure_ascii=False)
        
        return 'ok', eid
    finally:
        env.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data/json_2.1.0')
    p.add_argument('--output_dir', default='output')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    
    ep_files = sorted(glob.glob(os.path.join(args.output_dir, '*.json')))
    missing = sum(1 for f in ep_files if not os.path.isdir(f.replace('.json', '')))
    print(f'{len(ep_files)} episodes, {missing} missing screenshot dirs')
    
    tasks = [(f, args.data_dir, args.output_dir) for f in ep_files]
    
    stats = {'ok': 0, 'skip': 0, 'crash': 0}
    t0 = time.time()
    
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, t): t for t in tasks}
        done = 0
        for future in as_completed(futures):
            status, eid = future.result()
            stats[status] = stats.get(status, 0) + 1
            done += 1
            if done % 10 == 0:
                elapsed = time.time() - t0
                print(f'[{done}/{len(tasks)}] {status} {eid} [{done/elapsed:.1f}ep/s]', flush=True)
    
    elapsed = time.time() - t0
    print(f'Done {elapsed:.0f}s. OK={stats.get("ok",0)} SKIP={stats.get("skip",0)} CRASH={stats.get("crash",0)}')

if __name__ == '__main__':
    main()
