"""E4-C ordinary all-interface task controls; no claim of a novel optimizer."""
import argparse
import json
import time
import traceback
from pathlib import Path
import torch
import torch.nn.functional as F
from common import prepare


def bind_shared(modules, parameters, thresholds, timesteps):
    """One dimensionless parameter/channel, physically shared across T slots."""
    for name, p in parameters.items():
        modules[name].output_bias_correction = (p * thresholds[name]).view(1, 1, -1, 1, 1).expand(timesteps, 1, -1, 1, 1).contiguous()


def task_loss(logits, labels, teacher_logits, objective):
    if objective == 'ce':
        return F.cross_entropy(logits, labels)
    if objective == 'kl':
        return F.kl_div(F.log_softmax(logits, dim=1), F.softmax(teacher_logits, dim=1), reduction='batchmean')
    raise ValueError(objective)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--cpu-check', action='store_true'); args = p.parse_args()
    c = json.loads(Path(args.config).read_text()); begin = time.perf_counter(); r = None
    if args.cpu_check:
        receipt = prepare(c, cpu=True)
        assert c['fit_count'] % c['train_batch'] == 0
        assert c['epochs'] == 8 and c['checkpoints'] == [1, 4, 8]
        # Independent tiny surrogate gradient witness (no CUDA/real-image execution).
        from types import SimpleNamespace
        variables = {'x': torch.nn.Parameter(torch.zeros(3))}
        modules = {'x': SimpleNamespace()}
        bind_shared(modules, variables, {'x': 2.}, 2)
        assert torch.equal(modules['x'].output_bias_correction[0], modules['x'].output_bias_correction[1])
        modules['x'].output_bias_correction.sum().backward()
        assert torch.equal(variables['x'].grad, torch.full((3,), 4.))
        assert torch.isfinite(task_loss(torch.randn(4, 5), torch.arange(4), torch.randn(4, 5), 'kl'))
        from modules import ParaInfNeuron
        actual = {'first': ParaInfNeuron(2, th=1., init_mem=.5, dim=5),
                  'second': ParaInfNeuron(2, th=1., init_mem=.5, dim=5)}
        for m in actual.values():
            m.output_bias_correction = torch.zeros(2, 1, 3, 1, 1)
        q = {n: torch.nn.Parameter(torch.zeros(3)) for n in actual}
        bind_shared(actual, q, {n: 1. for n in actual}, 2)
        value = actual['second'](actual['first'](torch.full((4, 3, 2, 2), .51)))
        value.sum().backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum() > 0 for v in q.values())
        receipt['parameter_binding_gradient_passed'] = True
        receipt['actual_two_neuron_gradient_passed'] = True
        print(json.dumps(receipt)); return
    try:
        r = prepare(c); r.anchors()
        # One immutable image and teacher-logit cache shared by ordinary controls.
        batches = [(x.cuda(), y.cuda(), ids) for x, y, ids in r.e0.loader(r.rows['fit'], r.parent, c['train_batch'])]
        expected_ids = [row['index'] for row in r.rows['fit']]
        assert torch.equal(torch.cat([b[2] for b in batches]), torch.tensor(expected_ids))
        teacher_logits = []
        torch.cuda.synchronize(); teacher_cache_start = time.perf_counter()
        with torch.no_grad():
            for x, y, ids in batches:
                teacher_logits.append(r.e0.logits(r.teacher, x, r.parent).detach())
        torch.cuda.synchronize(); teacher_cache_seconds = time.perf_counter() - teacher_cache_start
        r.e0.atomic_torch(r.output / 'fit-teacher-logits.pt', {'ids': torch.cat([b[2] for b in batches]), 'logits': torch.cat(teacher_logits).cpu(), 'teacher_cache_seconds': teacher_cache_seconds})
        thresholds = r.bn['interface_thresholds']; T = r.parent['timesteps']; traces = {}
        r.progress('rate-ftbc-256fit')
        r.assign(r.bn['runtime_biases'])
        teacher_means, snn_means = {}, {}
        handles = []
        def hook_teacher(name):
            def f(_, inputs, output):
                teacher_means[name] = output.mean((0, 2, 3), keepdim=True).detach()
            return f
        def hook_snn(name):
            def f(_, inputs, output):
                rate = output.reshape(T, -1, *output.shape[1:]).mean(0)
                snn_means[name] = rate.mean((0, 2, 3), keepdim=True).detach()
            return f
        for name in r.names:
            handles.append(dict(r.teacher.named_modules())[name].register_forward_hook(hook_teacher(name)))
            handles.append(r.modules[name].register_forward_hook(hook_snn(name)))
        rate_trace = []; solver_seconds = 0.
        try:
            for epoch in range(1, c['epochs'] + 1):
                torch.cuda.synchronize(); started = time.perf_counter()
                with torch.no_grad():
                    for x, _, _ in batches:
                        teacher_means.clear(); snn_means.clear()
                        r.e0.logits(r.teacher, x, r.parent)
                        r.e0.logits(r.model, x, r.parent, True)
                        assert set(teacher_means) == set(snn_means) == set(r.names)
                        for name in r.names:
                            error = teacher_means[name] - snn_means[name]
                            r.modules[name].output_bias_correction.add_(c['rate_eta'] * error.unsqueeze(0).expand(T, *error.shape))
                            r.modules[name].output_bias_correction.clamp_(-c['bound_ratio'] * thresholds[name], c['bound_ratio'] * thresholds[name])
                torch.cuda.synchronize(); solver_seconds += time.perf_counter() - started
                rate_trace.append({'epoch': epoch, 'solver_seconds': solver_seconds})
                if epoch in c['checkpoints']:
                    # Hooks only collect batch means; no updates occur during score.
                    r.score(f'rate256-e{epoch}', r.snapshot(), {'objective': 'rate_mean', 'epoch': epoch, 'fit_images': c['fit_count'], 'solver_seconds': solver_seconds, 'eta': c['rate_eta']})
        finally:
            for h in handles: h.remove()
        traces['rate'] = rate_trace
        for objective in c['objectives']:
            for lr in c['lr_ratios']:
                r.assign(r.bn['runtime_biases'])
                params = {n: torch.nn.Parameter(torch.zeros(r.bn['runtime_biases'][n].shape[2], device='cuda')) for n in r.names}
                optimizer = torch.optim.Adam(list(params.values()), lr=lr, weight_decay=0.)
                trace = []; solver_seconds = 0.; first_gradient_norms = None
                for epoch in range(1, c['epochs'] + 1):
                    torch.cuda.synchronize(); started = time.perf_counter(); loss_sum = 0.; last_norm = 0.
                    for j, (x, y, _) in enumerate(batches):
                        bind_shared(r.modules, params, thresholds, T)
                        optimizer.zero_grad(set_to_none=True)
                        logits = r.e0.logits(r.model, x, r.parent, True)
                        loss = task_loss(logits, y, teacher_logits[j], objective)
                        assert torch.isfinite(loss)
                        loss.backward()
                        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in params.values())
                        if first_gradient_norms is None:
                            first_gradient_norms = {n: float(v.grad.norm()) for n, v in params.items()}
                        last_norm = float(torch.nn.utils.clip_grad_norm_(list(params.values()), 1.))
                        optimizer.step()
                        with torch.no_grad():
                            for v in params.values(): v.clamp_(-c['bound_ratio'], c['bound_ratio'])
                        loss_sum += float(loss.detach()) * len(y)
                    torch.cuda.synchronize(); solver_seconds += time.perf_counter() - started
                    trace.append({'epoch': epoch, 'minibatch_objective': loss_sum / c['fit_count'], 'last_gradient_norm': last_norm, 'first_step_interface_gradient_norms': first_gradient_norms, 'solver_seconds': solver_seconds})
                    r.progress('training', objective=objective, lr=lr, **trace[-1])
                    if epoch in c['checkpoints']:
                        bind_shared(r.modules, params, thresholds, T)
                        r.score(f'{objective}-lr{lr}-e{epoch}', r.snapshot(), {'objective': objective, 'lr_ratio': lr, 'epoch': epoch, 'solver_seconds': solver_seconds, 'fit_images': c['fit_count']})
                traces[f'{objective}-{lr}'] = trace
                r.assign(r.bn['runtime_biases'])
                del optimizer, params
        r.e0.atomic_json(r.output / 'traces.json', traces)
        result = r.finish(); result['whole_process_seconds'] = time.perf_counter() - begin
        result['teacher_cache_seconds'] = teacher_cache_seconds
        r.e0.atomic_json(r.output / 'result.json', result)
    except Exception:
        if r:
            r.e0.atomic_json(r.output / 'failure.json', {'traceback': traceback.format_exc(), 'elapsed': time.perf_counter()-begin})
        raise


if __name__ == '__main__':
    main()
