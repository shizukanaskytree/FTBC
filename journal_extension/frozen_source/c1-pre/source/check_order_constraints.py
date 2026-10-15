"""Exact CPU examples for local calibration invariants; no data or CUDA."""
from fractions import Fraction as F
from itertools import product, accumulate
import json
from pathlib import Path


def serial(path, bias, threshold=F(1), initial=F(1, 2)):
    membrane, count, trace = initial, 0, []
    for drive, correction in zip(path, bias):
        membrane += drive - correction
        spike = int(membrane >= threshold)
        membrane -= threshold * spike
        count += spike
        trace.append(count)
    return trace


def parallel(z, bias, threshold=F(1), initial=F(1, 2)):
    T = len(bias)
    return sum(T*z + initial + r*d >= r*threshold
               for r, d in enumerate(bias, 1))


def main():
    checks = 0
    for T in (2, 3):
        paths = list(product((F(-1), F(0), F(1)), repeat=T))
        biases = list(product((F(-1, 2), F(0), F(1, 2)), repeat=T))
        for x, y in product(paths, repeat=2):
            if not all(a <= b for a, b in zip(accumulate(x), accumulate(y))):
                continue
            for bias in biases:
                assert all(a <= b for a, b in zip(serial(x, bias), serial(y, bias)))
                checks += 1
    parallel_checks = 0
    for T in (2, 4):
        for bias in product((F(-1), F(0), F(2)), repeat=T):
            values = [parallel(F(i, 4), bias) for i in range(-8, 13)]
            assert values == sorted(values)
            parallel_checks += len(values)-1
    # Serial distinguishes crossing trajectories; final drive alone is insufficient.
    x, y, bias = (F(2), F(-1)), (F(0), F(3, 2)), (F(0), F(2))
    assert sum(x) < sum(y) and serial(x, bias)[-1] > serial(y, bias)[-1]
    grid_checks = 0
    for T in (2, 4):
        grid = [F(k, T) for k in range(T+1)]
        for a, b in product([F(i, 8) for i in range(-2, 11)], repeat=2):
            independent = min((v-a)**2 for v in grid) + min((v-b)**2 for v in grid)
            restricted = min((v-a)**2+(w-b)**2 for v, w in product(grid, repeat=2) if v <= w)
            penalty = restricted-independent
            assert penalty >= 0
            for v, w in product(grid, repeat=2):
                if v <= w:
                    assert (v-a)**2+(w-b)**2 >= independent+penalty
                    grid_checks += 1
    result = {
        'status': 'passed', 'arithmetic': 'exact fractions',
        'serial_prefix_order_cases': checks, 'parallel_adjacent_order_cases': parallel_checks,
        'grid_penalty_cases': grid_checks,
        'crossing_path_witness': {'x': ['2','-1'], 'y': ['0','3/2'],
          'subtracted_bias': ['0','2'], 'initial':'1/2', 'threshold':'1',
          'counts_x': serial(x,bias), 'counts_y': serial(y,bias)},
        'scope': 'Local fixed inputs, subtractive non-leaky IF, positive common threshold; no network or novelty conclusion.'}
    dest = Path(__file__).resolve().parents[1] / 'theory-checks.json'
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
