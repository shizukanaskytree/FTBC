"""E5 bounded diagnostics. Reuses E4 runtime without editing its frozen source."""
import argparse
import json
import sys
import time
import traceback
import resource
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / 'e4/source'))
from common import prepare
import torch


def load_bias(c, name):
    return torch.load(Path(c['e4_objective_output']) / ('bias-' + name + '.pt'), map_location='cpu', weights_only=False)['biases']


def means(y, T):
    return y.reshape(T, -1, *y.shape[1:]).mean(0).mean((0, 2, 3), keepdim=True)


@torch.no_grad()
def update(module, target, response, theta, c):
    error = target - response
    T = module.output_bias_correction.shape[0]
    module.output_bias_correction.add_(c['rate_eta'] * error.unsqueeze(0).expand(T, *error.shape))
    module.output_bias_correction.clamp_(-c['bound_ratio'] * theta, c['bound_ratio'] * theta)


def group_bias(r, biases, group):
    return {n: v.clone() if ((n.endswith('.relu')) == (group == 'post_add')) else torch.zeros_like(v) for n, v in biases.items()}


def vector_conditions(r, c):
    e1, kl = r.calibrated['runtime_biases'], load_bias(c, 'kl-lr0.001-e4')
    out = []
    generator = torch.Generator().manual_seed(c['seed'])
    shuffled = {n: v[:, :, torch.randperm(v.shape[2], generator=generator)].clone() for n, v in e1.items()}
    for scale in (1., .5):
        out.append((f'e1-shuffle-{scale}', {n: scale*v for n,v in shuffled.items()}, {'scale': scale, 'per_layer_channel_shuffle_seed': c['seed']}))
    for label, direction, norm_source in [('e1-norm-kl', e1, kl), ('kl-norm-e1', kl, e1)]:
        values, report = {}, {}
        for n in r.names:
            theta = r.bn['interface_thresholds'][n]
            norm = direction[n][0].double().norm().item()
            target = norm_source[n][0].double().norm().item()
            pre = direction[n] * (target / norm) if norm > 0 else torch.zeros_like(direction[n])
            value = pre.clamp(-c['bound_ratio']*theta, c['bound_ratio']*theta)
            values[n] = value
            report[n] = {'direction_norm': norm, 'requested_norm': target, 'actual_norm': value[0].double().norm().item(), 'clipped_channels': int((pre[0].abs() > c['bound_ratio']*theta).sum()), 'zero_direction': norm == 0}
        out.append((label, values, {'norms': report}))
    for label, b in [('e1', e1), ('kl-e4', kl)]:
        for group in ('direct_bn', 'post_add'):
            out.append((label+'-'+group, group_bias(r, b, group), {'group': group, 'deployment_only_not_refitted': True}))
    return out


@torch.no_grad()
def teacher_cache(r, batches):
    captured, handles = {}, []
    def hook(n):
        def fn(_, inputs, output): captured[n] = output.mean((0,2,3), keepdim=True).detach().clone()
        return fn
    for n in r.names: handles.append(dict(r.teacher.named_modules())[n].register_forward_hook(hook(n)))
    targets, logits = [], []
    try:
        for x, _, _ in batches:
            captured.clear()
            logits.append(r.e0.logits(r.teacher, x, r.parent).cpu())
            assert set(captured) == set(r.names)
            targets.append(dict(captured))
    finally:
        for h in handles: h.remove()
    r.e0.atomic_torch(r.output / 'fit-teacher-logits.pt', {'ids': torch.cat([b[2] for b in batches]), 'logits': torch.cat(logits)})
    return targets


@torch.no_grad()
def capture_inputs(r, batches, name, mode):
    """Cache immutable layer input. L uses teacher PRE-activation, not its output."""
    source = r.model if mode == 'G' else r.teacher
    module = dict(source.named_modules())[name]
    holder = {}
    def pre(_, inputs): holder['x'] = inputs[0].detach().clone()
    def post(_, inputs, output): holder['y'] = output.detach().clone()
    handles = [module.register_forward_pre_hook(pre), module.register_forward_hook(post)]
    cache = []
    T = r.parent['timesteps']
    try:
        for x, _, _ in batches:
            r.e0.logits(source, x, r.parent, mode == 'G')
            z = holder['x']
            if mode == 'L': z = z.repeat(T, 1, 1, 1)
            if mode == 'G':
                expected = holder['y']
                assert torch.equal(r.modules[name](z), expected), 'local replay differs before fitting'
            cache.append(z.cpu())
    finally:
        for h in handles: h.remove()
    return cache


@torch.no_grad()
def witness_after(r, x, z, name, mode):
    if mode != 'G': return
    holder = {}
    def pre(_, inputs): holder['x'] = inputs[0].detach().clone()
    def post(_, inputs, output): holder['y'] = output.detach().clone()
    handles = [r.modules[name].register_forward_pre_hook(pre), r.modules[name].register_forward_hook(post)]
    try: r.e0.logits(r.model, x, r.parent, True)
    finally:
        for h in handles: h.remove()
    assert torch.equal(holder['x'], z.cuda()), 'upstream changed during layer fit'
    assert torch.equal(holder['y'], r.modules[name](z.cuda())), 'local replay differs after fitting'


@torch.no_grad()
def fit_simultaneous(r, c, batches, targets):
    T = r.parent['timesteps']; current = {}; handles = []; trace = []
    def hook(n):
        def fn(_, inputs, output): current[n] = means(output, T).detach()
        return fn
    for n in r.names: handles.append(r.modules[n].register_forward_hook(hook(n)))
    r.assign(r.bn['runtime_biases']); start = time.perf_counter()
    try:
        for epoch in range(1, max(c['checkpoints'])+1):
            for j, (x, _, _) in enumerate(batches):
                current.clear(); r.e0.logits(r.model, x, r.parent, True)
                for n in r.names: update(r.modules[n], targets[j][n], current[n], r.bn['interface_thresholds'][n], c)
            if epoch in c['checkpoints']:
                b = r.snapshot()
                if c.get('check_e4',False):
                    reference = load_bias(c, f'rate256-e{epoch}')
                    assert all(torch.equal(b[n], reference[n]) for n in r.names), 'S must reproduce E4 exactly'
                elapsed = time.perf_counter()-start
                r.score(f'S-e{epoch}', b, {'epochs': epoch, 'updates_per_interface': epoch*len(batches), 'fit_seconds_including_intermediate_scores': elapsed, 'e4_bitwise_bias_match': c.get('check_e4',False)})
                trace.append({'epoch': epoch, 'updates_per_interface': epoch*len(batches)})
    finally:
        for h in handles: h.remove()
    return trace


@torch.no_grad()
def fit_layerwise(r, c, batches, targets, mode):
    T = r.parent['timesteps']; traces = []
    for epochs in c['checkpoints']:
        r.assign(r.bn['runtime_biases'])  # Independent restart for EACH per-layer budget.
        for name in r.names:
            start = time.perf_counter()
            cache = capture_inputs(r, batches, name, mode)
            for epoch in range(epochs):
                for j, z in enumerate(cache):
                    y = r.modules[name](z.cuda())
                    update(r.modules[name], targets[j][name], means(y,T), r.bn['interface_thresholds'][name], c)
            witness_after(r, batches[0][0], cache[0], name, mode)
            traces.append({'epochs': epochs, 'layer': name, 'updates': epochs*len(batches), 'capture_forwards': len(batches), 'local_update_forwards': epochs*len(batches), 'seconds': time.perf_counter()-start, 'before_witness_batches': len(batches) if mode=='G' else 0, 'after_witness_batches': int(mode=='G')})
            r.progress('layer_fitted', mode=mode, **traces[-1])
            del cache
        snapshot = r.snapshot()
        if c['sanity'] and mode == 'G':
            # Independently reproduce all updates with full model forward per update.
            r.assign(r.bn['runtime_biases'])
            for name in r.names:
                holder = {}
                def hook(_, inputs, output): holder['y'] = output.detach().clone()
                h = r.modules[name].register_forward_hook(hook)
                try:
                    for epoch in range(epochs):
                        for j, (x, _, _) in enumerate(batches):
                            r.e0.logits(r.model, x, r.parent, True)
                            update(r.modules[name], targets[j][name], means(holder['y'],T), r.bn['interface_thresholds'][name], c)
                finally: h.remove()
            assert all(torch.equal(r.snapshot()[n], snapshot[n]) for n in r.names), 'cached vs uncached G mismatch'
        r.score(f'{mode}-e{epochs}', snapshot, {'epochs_per_layer': epochs, 'updates_per_interface': epochs*len(batches), 'independent_zero_restart': True, 'cached_vs_uncached_checked': c['sanity'] and mode=='G'})
    return traces


def cpu_check(c):
    check = prepare(c, cpu=True)
    assert not check['cuda_initialized']
    assert c['fit_count'] % c['train_batch'] == 0
    names = check['interfaces']; assert sum(n.endswith('.relu') for n in names)==16
    assert c['rate_eta']==.01 and c['bound_ratio']==.5
    from modules import ParaInfNeuron
    T = 2
    m = ParaInfNeuron(T, th=1., init_mem=.5, dim=5)
    m.output_bias_correction = torch.zeros(T,1,3,1,1)
    x = torch.linspace(-.2,1.2,48).reshape(4,3,2,2)
    z = x[:2].repeat(T,1,1,1)
    y = m(z)
    target = torch.full((1,3,1,1), .7)
    update(m, target, means(y,T), 1., c)
    assert torch.equal(m.output_bias_correction[0],m.output_bias_correction[1])
    assert torch.isfinite(m(z)).all()
    # Saturation: bound is theta-relative, with both signs.
    update(m, torch.full_like(target,1000.), means(m(z),T),1.,c)
    assert torch.equal(m.output_bias_correction,torch.full_like(m.output_bias_correction,.5))
    update(m, torch.full_like(target,-1000.), means(m(z),T),1.,c)
    assert torch.equal(m.output_bias_correction,torch.full_like(m.output_bias_correction,-.5))
    check['actual_neuron_and_bounds_passed']=True
    return check


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--cpu-check',action='store_true'); a=p.parse_args()
    c=json.loads(Path(a.config).read_text()); begin=time.perf_counter(); r=None
    if a.cpu_check: print(json.dumps(cpu_check(c))); return
    try:
        r=prepare(c)
        r.score('bn-only',r.bn['runtime_biases'])
        r.score('ftbc-e1',r.calibrated['runtime_biases'],{'fit_images':4096,'unbounded_historical_reference':True})
        r.score('kl-e4',load_bias(c,'kl-lr0.001-e4'),{'fit_images':256,'selected_on_old_development':True})
        r.score('e1-scale-0.125',{n:v*.125 for n,v in r.calibrated['runtime_biases'].items()},{'selected_on_old_development':True})
        batches=[(x.cuda(),y,ids) for x,y,ids in r.e0.loader(r.rows['fit'],r.parent,c['train_batch'])]
        assert torch.equal(torch.cat([b[2] for b in batches]),torch.tensor([x['index'] for x in r.rows['fit']]))
        start=time.perf_counter(); targets=teacher_cache(r,batches); torch.cuda.synchronize(); cache_seconds=time.perf_counter()-start
        if c['mode']=='vector':
            for ident,b,detail in vector_conditions(r,c): r.score(ident,b,detail)
            trace=[]
        elif c['mode']=='S': trace=fit_simultaneous(r,c,batches,targets)
        else: trace=fit_layerwise(r,c,batches,targets,c['mode'])
        if c['mode']!='vector' and not c['sanity']:
            full=r.snapshot()
            for group in ('direct_bn','post_add'):
                r.score(f'{c["mode"]}-e8-{group}',group_bias(r,full,group),{'deployment_only_not_refitted':True})
        r.e0.atomic_json(r.output/'traces.json',trace)
        result=r.finish()
        result['process_peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        cg=Path('/proc/self/cgroup').read_text().strip().split('::')[-1]
        peak=Path('/sys/fs/cgroup')/cg.lstrip('/')/'memory.peak'
        result['task_cgroup_memory_peak_bytes']=int(peak.read_text()) if peak.exists() else None
        result.update(whole_process_seconds=time.perf_counter()-begin,teacher_cache_seconds=cache_seconds,data_status='prospective_research_internal_development_not_confirmation',sanity=c['sanity'])
        r.e0.atomic_json(r.output/'result.json',result)
    except Exception:
        if r: r.e0.atomic_json(r.output/'failure.json',{'traceback':traceback.format_exc(),'seconds':time.perf_counter()-begin})
        raise

if __name__=='__main__': main()
