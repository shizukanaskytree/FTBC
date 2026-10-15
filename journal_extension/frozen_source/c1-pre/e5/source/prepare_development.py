"""Create a prospective train-derived development whitelist; exclude protected IDs/content."""
import hashlib
import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np

ROOT=Path('/data3/snn_prjs/ftbc_neural_networks')
E=Path(__file__).resolve().parents[1]
BASE=ROOT/'ccfa-workfiles/experiments/ftbc-neural-networks'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,x):
    assert not p.exists(),p
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def ints(x):
    if isinstance(x,dict):return set().union(*(ints(v) for v in x.values())) if x else set()
    if isinstance(x,list):
        if x and all(isinstance(v,int) for v in x):return set(x)
        return set().union(*(ints(v) for v in x)) if x else set()
    return set()
def hashes(x):
    if isinstance(x,dict):return set().union(*(hashes(v) for v in x.values())) if x else set()
    if isinstance(x,list):return set().union(*(hashes(v) for v in x)) if x else set()
    return {x} if isinstance(x,str) and re.fullmatch('[0-9a-f]{64}',x) else set()

paths=[BASE/'parallel-main-inputs.json',BASE/'protocol-inputs.json',BASE/'h004-development-exposure-01.json',BASE/'ssd-runs/FTBC-R054-CALIBRATION-SAMPLE-PREPARE-01/preparation.json',E.parent/'input-roles.json']
main,old,exposure,stability,e0=[json.loads(p.read_text()) for p in paths]
roles=main['datasets']['ImageNet']['roles']
excluded=ints(roles['calibration'])|ints(roles['development'])|ints(roles['confirmation'])
excluded|=ints(old['settings']['imagenet_relu_r18_t4'])|set(exposure['exclude_from_future_study_confirmation']['imagenet_relu_r18_t4'])
excluded|={r['index'] for rows in e0.values() for r in rows}
excluded_sha=hashes(stability['settings']['imagenet'])|{r['sha256'] for rows in e0.values() for r in rows}
inventory=ROOT/main['datasets']['ImageNet']['splits']['train']['records_path']
assert sha(inventory)==main['datasets']['ImageNet']['splits']['train']['records_sha256']
order=np.random.Generator(np.random.PCG64(2026100905)).permutation(main['datasets']['ImageNet']['splits']['train']['count'])
candidates=[int(i) for i in order if int(i) not in excluded][:2048];want=set(candidates);rows={}
for line in inventory.open():
    index=int(re.search(r'"index"\s*:\s*(\d+)',line).group(1))
    if index in excluded:
        # Only identity metadata is used for protected-role exclusion. No protected image or prediction payload.
        excluded_sha.add(re.search(r'"sha256"\s*:\s*"([0-9a-f]{64})"',line).group(1))
    elif index in want:rows[index]=json.loads(line)
chosen=[];seen=set()
for i in candidates:
    row=rows[i]
    if row['sha256'] in excluded_sha or row['sha256'] in seen:continue
    chosen.append(dict(row,path=str((ROOT/'datasets/imagenet/train'/row['path']).resolve())))
    seen.add(row['sha256'])
    if len(chosen)==512:break
assert len(chosen)==512
def check(r):assert sha(r['path'])==r['sha256'],r['index']
with ThreadPoolExecutor(8) as pool:list(pool.map(check,chosen))
export={'fit':e0['fit'],'diagnostic':e0['diagnostic'],'development':chosen}
save(E/'input-roles.json',export)
parent=json.loads((E.parent/'config/parallel.json').read_text())
parent.update(dataset_manifest=str(E/'input-roles.json'),dataset_manifest_sha256=sha(E/'input-roles.json'))
save(E/'parent-config.json',parent)
save(E/'input-preparation.json',{'passed':True,'role':'prospective research-internal development, not confirmation','seed':2026100905,'fresh_development_count':512,'fit_inherited_count':4096,'excluded_ids':len(excluded),'excluded_content_hashes':len(excluded_sha),'exclusion_sources':{str(p):sha(p) for p in paths},'new_manifest_sha256':sha(E/'input-roles.json'),'protected_image_label_prediction_payload_opened':False,'protected_ID_hash_metadata_used_for_exclusion':True,'image_hashes_verified':True,'historical_exposure':'Source ANN and historical full training may have seen these training images; no historically pristine claim. Excluded known current study calibration/development/confirmation and prior recorded pilot content.'})
print(json.dumps({'passed':True,'new_development':512,'excluded_ids':len(excluded),'excluded_hashes':len(excluded_sha)}))
