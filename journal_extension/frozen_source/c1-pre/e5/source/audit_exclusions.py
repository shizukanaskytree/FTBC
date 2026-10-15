"""CPU addendum: per-source exclusion counts and independent zero-overlap checks."""
import json,re,hashlib
from pathlib import Path
E=Path(__file__).resolve().parents[1];ROOT=E.parents[5]
BASE=ROOT/'ccfa-workfiles/experiments/ftbc-neural-networks'
# Resolve workspace explicitly; this script may be moved only under a new freeze.
ROOT=Path('/data3/snn_prjs/ftbc_neural_networks');BASE=ROOT/'ccfa-workfiles/experiments/ftbc-neural-networks'
main=json.loads((BASE/'parallel-main-inputs.json').read_text())
old=json.loads((BASE/'protocol-inputs.json').read_text())['settings']['imagenet_relu_r18_t4']
h004=json.loads((BASE/'h004-development-exposure-01.json').read_text())['exclude_from_future_study_confirmation']['imagenet_relu_r18_t4']
stability=json.loads((BASE/'ssd-runs/FTBC-R054-CALIBRATION-SAMPLE-PREPARE-01/preparation.json').read_text())['settings']['imagenet']
e0=json.loads((E.parent/'input-roles.json').read_text());fresh=json.loads((E/'input-roles.json').read_text())['development']
def ids(v):
    if isinstance(v,dict): return set().union(*(ids(x) for x in v.values())) if v else set()
    if isinstance(v,list):
        if all(isinstance(x,int) and not isinstance(x,bool) for x in v):return set(v)
        return set().union(*(ids(x) for x in v)) if v else set()
    return set()
def hashes(v):
    if isinstance(v,dict):return set().union(*(hashes(x) for x in v.values())) if v else set()
    if isinstance(v,list):return set().union(*(hashes(x) for x in v)) if v else set()
    return {v} if isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) else set()
roles=main['datasets']['ImageNet']['roles'];sources={}
for role in ['calibration','development','confirmation']:
    sources['main_'+role]={'ids':ids(roles[role]),'hashes':set()}
assert isinstance(h004,list) and h004 and all(isinstance(x,int) and not isinstance(x,bool) for x in h004)
sources['old_protocol']={'ids':ids(old),'hashes':set()};sources['h004']={'ids':set(h004),'hashes':set()}
sources['E0']={'ids':{r['index'] for rows in e0.values() for r in rows},'hashes':{r['sha256'] for rows in e0.values() for r in rows}}
sources['stability']={'ids':set(),'hashes':hashes(stability)}
for name,v in sources.items():assert v['ids'] or v['hashes'],name
union=set().union(*(v['ids'] for v in sources.values()));id_hash={}
p=ROOT/main['datasets']['ImageNet']['splits']['train']['records_path']
for line in p.open():
    i=int(re.search(r'"index"\s*:\s*(\d+)',line).group(1))
    if i in union:id_hash[i]=re.search(r'"sha256"\s*:\s*"([0-9a-f]{64})"',line).group(1)
fresh_ids={r['index'] for r in fresh};fresh_hashes={r['sha256'] for r in fresh};report={}
for name,v in sources.items():
    assert all(i in id_hash for i in v['ids']),name
    h=v['hashes']|{id_hash[i] for i in v['ids']}
    assert not fresh_ids & v['ids'] and not fresh_hashes & h,name
    report[name]={'excluded_IDs':len(v['ids']),'excluded_hashes':len(h),'new_development_ID_overlap':0,'new_development_content_overlap':0,'nonempty_source':True}
(E/'exclusion-audit.json').write_text(json.dumps({'passed':True,'sources':report,'fresh_count':512,'protected_scope':'ID/hash metadata only for exclusions; no protected image files or prediction payload'},indent=2)+'\n')
print(json.dumps(report))
