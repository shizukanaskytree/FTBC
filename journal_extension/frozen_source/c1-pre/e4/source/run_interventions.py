"""Predeclared deterministic interventions on the E1 bias, not a new method."""
import argparse
import json
import time
import traceback
from pathlib import Path
import torch
from common import prepare


def interventions(names, mode):
    if mode == 'scale':
        for value in [-1., -.5, .125, .25, .5, .75, 1.25]:
            yield 'scale-' + str(value), {n: value for n in names}
        for group in ['conv1', 'conv2_x', 'conv3_x', 'conv4_x', 'conv5_x']:
            for kind in ['only', 'omit']:
                selected = {n for n in names if n.split('.')[0] == group}
                yield kind + '-' + group, {n: float((n in selected) == (kind == 'only')) for n in names}
    elif mode == 'layer':
        for i, name in enumerate(names):
            yield f'only-layer-{i:02}', {n: float(n == name) for n in names}
            yield f'omit-layer-{i:02}', {n: float(n != name) for n in names}
    else:
        raise ValueError(mode)


def main():
    a = argparse.ArgumentParser(); a.add_argument('--config', required=True); a.add_argument('--cpu-check', action='store_true'); args = a.parse_args()
    c = json.loads(Path(args.config).read_text()); started = time.perf_counter()
    if args.cpu_check:
        receipt = prepare(c, cpu=True)
        cases = list(interventions(receipt['interfaces'], c['mode']))
        assert len({n for n, _ in cases}) == len(cases)
        assert all(set(v) == set(receipt['interfaces']) for _, v in cases)
        receipt.update(interventions=len(cases), total_candidates=2+len(cases))
        print(json.dumps(receipt)); return
    runtime = None
    try:
        runtime = prepare(c); runtime.anchors()
        cases = list(interventions(runtime.names, c['mode']))
        if c.get('sanity'):
            cases = cases[:2]
        for ident, factors in cases:
            biases = {n: runtime.calibrated['runtime_biases'][n] * factors[n] for n in runtime.names}
            runtime.score(ident, biases, {'factors': factors}, diagnostics=True)
        result = runtime.finish()
        result['whole_process_seconds'] = time.perf_counter() - started
        runtime.e0.atomic_json(runtime.output / 'result.json', result)
    except Exception:
        if runtime:
            runtime.e0.atomic_json(runtime.output / 'failure.json', {'traceback': traceback.format_exc(), 'elapsed': time.perf_counter()-started})
        raise


if __name__ == '__main__':
    main()
