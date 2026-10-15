"""Run at most three own-allocation children. Slurm owns device visibility."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--configs',nargs='+',required=True); a=p.parse_args()
assert os.environ.get('SLURM_JOB_ID') and 'slurm' in Path('/proc/self/cgroup').read_text().lower()
assert 1 <= len(a.configs) <= 3
children=[]
for config in a.configs:
    out=Path(json.loads(Path(config).read_text())['output'])
    assert not any((out/n).exists() for n in ['child-console.txt','progress.json','result.json','failure.json'])
for config in a.configs:
    c=json.loads(Path(config).read_text()); out=Path(c['output']); out.mkdir(parents=True,exist_ok=True)
    assert not (out/'child-console.txt').exists()
    log=(out/'child-console.txt').open('w')
    command=[sys.executable,'-B',str(Path(__file__).with_name('measure.py')),'--config',config]
    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    children.append((child,log,config))
# Wait for all children; a failure does not kill an independent sibling's evidence.
statuses=[]
for child,log,config in children:
    code=child.wait(); log.close(); statuses.append({'config':config,'pid':child.pid,'returncode':code})
print(json.dumps({'job':os.environ['SLURM_JOB_ID'],'children':statuses}),flush=True)
sys.exit(int(any(x['returncode'] for x in statuses)))
