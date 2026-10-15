"""E20 Slurm payload (EXPERIMENT_PLAN.md section 8): two child chains run concurrently on the one H100 of this allocation (the author's
2026-10-10 authorization: run in parallel as long as the GPU does not OOM; GPU_SCHEDULING.md: several child processes on the project's own card).
Each chain is sanity -> its frozen CPU acceptance -> main, as in E18/E19; main starts only after the sanity run completed and its acceptance
passed. One chain's failure does not stop the other; the job exits nonzero if either failed.
Both chains run with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (allocator only, no numerical effect) and PYTHONPATH with memwatch/
(a read-only exit hook that records each CUDA process's allocator counters to PAR_MEMSTATS). Device memory of the allocated H100 is sampled
with nvidia-smi every 0.5 s (total) and every 1 s (per process) for the whole job."""
import json, os, subprocess, sys, threading, time
from pathlib import Path
sys.dont_write_bytecode = True
E = Path(__file__).resolve().parents[1]
PY = '/home/wxf/miniconda3/envs/snn/bin/python'
H100 = 'GPU-4bfc38ec-7b5c-032d-993e-16750f4fe211'


def chain(model):
    return [(f'{model}-sanity', [PY, '-B', str(E / f'source/e20_{model}_runner.py'), '--config', str(E / f'config/{model}-sanity.json')]),
            (f'{model}-sanity-acceptance', [PY, '-B', str(E / f'source/e20_{model}_accept.py'), 'sanity']),
            (f'{model}-main', [PY, '-B', str(E / f'source/e20_{model}_runner.py'), '--config', str(E / f'config/{model}-main.json')])]


def samplers():
    return [('gpumem', ['nvidia-smi', '-i', H100, '--query-gpu=timestamp,uuid,memory.used,memory.total', '--format=csv,noheader,nounits', '-lms', '500']),
            ('gpuapps', ['nvidia-smi', '-i', H100, '--query-compute-apps=timestamp,gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits', '-lms', '1000'])]


def run_chain(model, env, job, codes, lock):
    log = E / 'slurm' / f'{model}-{job}.out'
    with log.open('w') as fh:
        for name, cmd in chain(model):
            if name.endswith('-main'):
                acc = E / f'{model}-sanity-acceptance.json'
                ok = acc.exists() and (lambda a: a['passed'] and not a['cuda_initialized'] and a['scope'] == 'sanity only')(json.loads(acc.read_text()))
                if not ok:
                    with lock: print(json.dumps({'stage': name, 'skipped': 'sanity acceptance did not pass'}), flush=True)
                    codes[model] = 3; return
            with lock: print(json.dumps({'stage': name, 'command': cmd, 'unix_time': time.time()}), flush=True)
            fh.write(json.dumps({'stage': name, 'command': cmd}) + '\n'); fh.flush()
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
            with lock: print(json.dumps({'stage': name, 'returncode': r.returncode, 'unix_time': time.time()}), flush=True)
            if r.returncode != 0:
                codes[model] = r.returncode; return
    codes[model] = 0


def main():
    job = os.environ.get('SLURM_JOB_ID', 'none')
    for model in ['r34', 'vgg']:
        assert not (E / f'{model}-sanity-acceptance.json').exists(), f'stale {model} sanity acceptance; reconcile before running'
        assert not (E / 'slurm' / f'{model}-{job}.out').exists(), f'stale {model} child log; reconcile before running'
    memstats = E / 'slurm' / f'memstats-{job}.jsonl'; assert not memstats.exists(), f'stale {memstats}; reconcile before running'
    path = os.pathsep.join([str(E / 'memwatch')] + ([os.environ['PYTHONPATH']] if os.environ.get('PYTHONPATH') else []))
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', PYTHONPATH=path, PAR_MEMSTATS=str(memstats))
    print(json.dumps({'stage': 'environment', 'PYTORCH_CUDA_ALLOC_CONF': env['PYTORCH_CUDA_ALLOC_CONF'], 'PYTHONPATH': env['PYTHONPATH'],
                      'PAR_MEMSTATS': env['PAR_MEMSTATS'], 'CUDA_VISIBLE_DEVICES': env.get('CUDA_VISIBLE_DEVICES'), 'unix_time': time.time()}), flush=True)
    watch = []
    for name, cmd in samplers():
        log = E / 'slurm' / f'{name}-{job}.csv'; assert not log.exists(), f'stale {log}'
        fh = log.open('w'); watch.append((subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT), fh))
    codes, lock = {}, threading.Lock()
    try:
        threads = [threading.Thread(target=run_chain, args=(m, env, job, codes, lock)) for m in ['r34', 'vgg']]
        for t in threads: t.start()
        for t in threads: t.join()
    finally:
        for w, fh in watch:
            w.terminate()
            try:
                w.wait(timeout=10)
            except subprocess.TimeoutExpired:
                w.kill(); w.wait()
            fh.close()
    print(json.dumps({'stage': 'all-finished', 'returncodes': codes}), flush=True)
    if any(codes.get(m, 1) for m in ['r34', 'vgg']):
        raise SystemExit(f'E20: a chain failed: {codes}')


if __name__ == '__main__':
    main()
