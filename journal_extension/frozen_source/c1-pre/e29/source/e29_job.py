"""E29 Slurm payload (EXPERIMENT_PLAN.md section 7; the E23 driver with the memwatch hook removed): the R34 chain and then the VGG-16 chain on the
one H100 of this allocation, one process at a time. Each chain is sanity -> its frozen CPU acceptance -> main; main starts only after the sanity run
completed and its acceptance's gate held (valid, and the paired checks P1 and P2). A valid sanity run whose gate did not hold ends the chain
without main as a pre-registered result (follow-up 1 N1). One chain's failure does not stop the other; the job exits nonzero if either failed. Both chains run with
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (allocator only, no numerical effect). The allocated H100 is sampled with nvidia-smi for the
whole job (descriptive: every 0.5 s the device, every 1 s the compute processes)."""
import json, os, subprocess, sys, threading, time
from pathlib import Path
sys.dont_write_bytecode = True
E = Path(__file__).resolve().parents[1]
PY = '/home/wxf/miniconda3/envs/snn/bin/python'
H100 = 'GPU-4bfc38ec-7b5c-032d-993e-16750f4fe211'


def chain(model):
    return [(f'{model}-sanity', [PY, '-B', str(E / 'source/e29_runner.py'), '--config', str(E / f'config/{model}-sanity.json')]),
            (f'{model}-sanity-acceptance', [PY, '-B', str(E / 'source/e29_accept.py'), model, 'sanity']),
            (f'{model}-main', [PY, '-B', str(E / 'source/e29_runner.py'), '--config', str(E / f'config/{model}-main.json')])]


def samplers():
    return [('gpumem', ['nvidia-smi', '-i', H100, '--query-gpu=timestamp,uuid,memory.used,memory.total,clocks.sm,power.draw,temperature.gpu',
                        '--format=csv,noheader,nounits', '-lms', '500']),
            ('gpuapps', ['nvidia-smi', '-i', H100, '--query-compute-apps=timestamp,gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits', '-lms', '1000'])]


def run_chain(model, env, job, codes, lock):
    log = E / 'slurm' / f'{model}-{job}.out'
    with log.open('w') as fh:
        for name, cmd in chain(model):
            if name.endswith('-main'):
                acc = E / f'{model}-sanity-acceptance.json'
                ok = acc.exists() and (lambda a: a['gate'] and a['valid'] and not a['cuda_initialized'] and a['scope'] == 'sanity only')(json.loads(acc.read_text()))
                if not ok:
                    with lock: print(json.dumps({'stage': name, 'skipped': 'the sanity gate did not hold'}), flush=True)
                    codes[model] = 3; return
            with lock: print(json.dumps({'stage': name, 'command': cmd, 'unix_time': time.time()}), flush=True)
            fh.write(json.dumps({'stage': name, 'command': cmd}) + '\n'); fh.flush()
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
            with lock: print(json.dumps({'stage': name, 'returncode': r.returncode, 'unix_time': time.time()}), flush=True)
            if r.returncode != 0:
                acc = E / f'{model}-sanity-acceptance.json'
                a = json.loads(acc.read_text()) if name.endswith('-sanity-acceptance') and acc.exists() else None
                if a and a['valid'] and not a['gate'] and not a['cuda_initialized'] and a['scope'] == 'sanity only':
                    with lock: print(json.dumps({'stage': name, 'gate_not_held': True, 'main': 'not run (a pre-registered result)'}), flush=True)
                    codes[model] = 0; return                                # follow-up 1 N1: the chain is complete
                codes[model] = r.returncode; return
    codes[model] = 0


def main(sequential=True):
    job = os.environ.get('SLURM_JOB_ID', 'none')
    for model in ['r34', 'vgg']:
        assert not (E / f'{model}-sanity-acceptance.json').exists(), f'stale {model} sanity acceptance; reconcile before running'
        assert not (E / 'slurm' / f'{model}-{job}.out').exists(), f'stale {model} child log; reconcile before running'
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    print(json.dumps({'stage': 'environment', 'PYTORCH_CUDA_ALLOC_CONF': env['PYTORCH_CUDA_ALLOC_CONF'], 'CUDA_VISIBLE_DEVICES': env.get('CUDA_VISIBLE_DEVICES'),
                      'sequential': sequential, 'unix_time': time.time()}), flush=True)
    watch = []
    for name, cmd in samplers():
        log = E / 'slurm' / f'{name}-{job}.csv'; assert not log.exists(), f'stale {log}'
        fh = log.open('w'); watch.append((subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT), fh))
    codes, lock = {}, threading.Lock()
    try:
        if sequential:
            for m in ['r34', 'vgg']:
                run_chain(m, env, job, codes, lock)
        else:
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
        raise SystemExit(f'E29: a chain failed: {codes}')


if __name__ == '__main__':
    assert sys.argv[1:] == [], sys.argv
    main(sequential=True)
