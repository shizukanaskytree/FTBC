"""E19 Slurm payload (same design as E18) (EXPERIMENT_PLAN.md section 8): sanity, its frozen CPU acceptance, then main, in one allocation.
The A100 queue is shared with long research-tier jobs. A separate main submission after the sanity acceptance can lose the card for hours,
so the same gate runs inside the allocation: main starts only after the sanity run completed and its acceptance passed."""
import json, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode = True
E = Path(__file__).resolve().parents[1]
PY = '/home/wxf/miniconda3/envs/snn/bin/python'


def stages():
    return [('sanity', [PY, '-B', str(E / 'source/e19_runner.py'), '--config', str(E / 'config/sanity.json')]),
            ('sanity-acceptance', [PY, '-B', str(E / 'source/e19_accept.py'), 'sanity']),
            ('main', [PY, '-B', str(E / 'source/e19_runner.py'), '--config', str(E / 'config/main.json')])]


def main():
    acceptance = E / 'sanity-acceptance.json'
    assert not acceptance.exists(), 'stale sanity acceptance; reconcile before running'
    for name, cmd in stages():
        if name == 'main':
            acc = json.loads(acceptance.read_text())
            assert acc['passed'] and not acc['cuda_initialized'] and acc['scope'] == 'sanity only', 'sanity acceptance did not pass; main not started'
        print(json.dumps({'stage': name, 'command': cmd}), flush=True)
        r = subprocess.run(cmd)
        print(json.dumps({'stage': name, 'returncode': r.returncode}), flush=True)
        if r.returncode != 0:
            raise SystemExit(f'E19 stage {name} failed ({r.returncode}); later stages not started')


if __name__ == '__main__':
    main()
