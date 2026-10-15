"""Locked evaluation v2 Slurm payload (plan section 5; adapted from the E25 driver): one child chain per job, sanity -> its frozen CPU acceptance
-> main; main starts only after the sanity run completed and its acceptance passed. Usage: l1_job.py <job tag>. One process on the card at a
time. The allocated GPU is sampled with nvidia-smi every 1 s for the record (device memory, and the compute processes, since a shared card
can carry other jobs). The main acceptance and the analysis run after the job, on the CPU."""
import json, os, subprocess, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
E = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(E / 'source'))
PY = '/home/wxf/miniconda3/envs/snn/bin/python'
IMAGENET = [f'{m}-t{T}' for m in ['r34', 'vgg'] for T in (2, 4, 8)]
CIFAR = ['c10-vgg16', 'c10-resnet20', 'c100-vgg16', 'c100-resnet20']


def chain(tag):
    runner = 'l1_imagenet_runner.py' if tag in IMAGENET else 'l1_cifar_runner.py'
    return [(f'{tag}-sanity', [PY, '-B', str(E / f'source/{runner}'), '--config', str(E / f'config/{tag}-sanity.json')]),
            (f'{tag}-sanity-acceptance', [PY, '-B', str(E / 'source/l1_accept.py'), tag, 'sanity']),
            (f'{tag}-main', [PY, '-B', str(E / f'source/{runner}'), '--config', str(E / f'config/{tag}-main.json')])]


def main(tag):
    job = os.environ.get('SLURM_JOB_ID', 'none')
    assert not (E / f'{tag}-sanity-acceptance.json').exists(), f'stale {tag} sanity acceptance; reconcile before running'
    log = E / 'slurm' / f'{tag}-{job}.out'; assert not log.exists(), f'stale {log}'
    gm = E / 'slurm' / f'gpumem-{tag}-{job}.csv'; gp = E / 'slurm' / f'gpumem-procs-{tag}-{job}.csv'
    assert not gm.exists() and not gp.exists(), 'stale sampler logs'
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    print(json.dumps({'stage': 'environment', 'tag': tag, 'CUDA_VISIBLE_DEVICES': env.get('CUDA_VISIBLE_DEVICES'), 'unix_time': time.time()}), flush=True)
    fh_gm = gm.open('w'); sampler = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,uuid,memory.used,memory.total', '--format=csv,noheader,nounits', '-lms', '1000'],
                                                       stdout=fh_gm, stderr=subprocess.STDOUT)
    fh_gp = gp.open('w'); procs = subprocess.Popen(['nvidia-smi', '--query-compute-apps=timestamp,gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits', '-lms', '1000'],
                                                   stdout=fh_gp, stderr=subprocess.STDOUT)
    code = 0
    try:
        with log.open('w') as fh:
            for name, cmd in chain(tag):
                if name.endswith('-main'):
                    FZ = __import__('l1_freeze_submit'); FZ.approved()       # initial review N9: the author's approval of the current plan and freeze
                    acc = E / f'{tag}-sanity-acceptance.json'
                    ok = acc.exists() and (lambda a: a['passed'] and not a['cuda_initialized'] and a['scope'] == 'sanity only')(json.loads(acc.read_text()))
                    if not ok:
                        print(json.dumps({'stage': name, 'skipped': 'sanity acceptance did not pass'}), flush=True); code = 3; break
                print(json.dumps({'stage': name, 'command': cmd, 'unix_time': time.time()}), flush=True)
                fh.write(json.dumps({'stage': name, 'command': cmd}) + '\n'); fh.flush()
                r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
                print(json.dumps({'stage': name, 'returncode': r.returncode, 'unix_time': time.time()}), flush=True)
                if r.returncode != 0:
                    code = r.returncode; break
    finally:
        for s in (sampler, procs):
            s.terminate()
            try:
                s.wait(timeout=10)
            except subprocess.TimeoutExpired:
                s.kill(); s.wait()
        fh_gm.close(); fh_gp.close()
    print(json.dumps({'stage': 'finished', 'tag': tag, 'returncode': code}), flush=True)
    if code:
        raise SystemExit(f'locked evaluation {tag}: the chain failed with {code}')


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in IMAGENET + CIFAR, sys.argv
    main(sys.argv[1])
