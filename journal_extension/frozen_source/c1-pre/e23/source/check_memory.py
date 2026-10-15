"""E23 post-run memory record (EXPERIMENT_PLAN.md section 7, memory record; the E20/E22 check with the E23 runner names), on CPU. Usage:
check_memory.py <slurm job id>. Writes slurm/memory-check.json. Descriptive in E23 (initial review B3): the chains run one after the other,
so the record never decides the validity of the other model, and a release-default OOM is a recorded outcome of that arm.
Pre-registered reading: every runner process that started (its stage line is in the chain log) must have left exactly one exit record,
otherwise its memory state is unknown. For a process that used the GPU, its num_ooms is compared with the increases its run's cost.json
recorded per arm (follow-up 1 B5: every timed arm records the increase of num_ooms during the call, and an arm recorded as out_of_memory the
increase during its attempt). Flagged in the results next to the cost table: OOMs outside every arm (total minus the recorded increases, e.g.
during scoring); completed arms with an increase (they finished after a caught allocation failure, e.g. a cuDNN algorithm fallback, so their
timing may be affected); out_of_memory arms without an increase (the error was raised outside the caching allocator). num_alloc_retries and the device samples (peak used, minimum free) are reported; retries > 0 or a minimum free below 1 GiB
sets near_full, flagged likewise. The exit-time peaks are peaks since the last per-arm reset, not process peaks (initial review N9)."""
import csv, json, sys
from pathlib import Path
E = Path(__file__).resolve().parents[1]
GIB = 2 ** 30


def stages(log):
    out = []
    for line in log.read_text(errors='replace').splitlines() if log.exists() else []:
        if line.startswith('{"stage"'):
            try:
                out.append(json.loads(line)['stage'])
            except (ValueError, KeyError):              # a runner line cut by a kill; never a chain stage line
                pass
    return out


def expected(job, slurm):
    out = {}
    for model in ['r34', 'vgg']:
        st = stages(slurm / f'{model}-{job}.out')
        for which in ['sanity', 'main']:
            if f'{model}-{which}' in st:
                out[f'{model}-{which}'] = (E / f'source/e23_{model}_runner.py', E / f'config/{model}-{which}.json')
    return out


def recorded_ooms(cfg):
    """The num_ooms increases recorded in the run's cost.json: out_of_memory arms and completed arms with a nonzero increase."""
    f = Path(json.loads(Path(cfg).read_text())['output']) / 'cost.json'
    cost = json.loads(f.read_text()) if f.exists() else {}
    return {'cost_json': f.exists(), 'oom_arms': {a: v.get('num_ooms_increase', 0) for a, v in cost.items() if v.get('status') == 'out_of_memory'},
            'completed_arms': {a: v['num_ooms_increase'] for a, v in cost.items() if v.get('status', 'completed') == 'completed' and v.get('num_ooms_increase')}}


def check(job, slurm=E / 'slurm', recorded=None):
    f = slurm / f'memstats-{job}.jsonl'
    recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []
    exp = expected(job, slurm); found = {}
    for key, (runner, cfg) in exp.items():
        found[key] = [r for r in recs if len(r['argv']) >= 3 and Path(r['argv'][0]).resolve() == runner.resolve() and r['argv'][1:3] == ['--config', str(cfg)]]
    unexpected = [r for r in recs if not any(r in v for v in found.values())]
    complete = all(len(v) == 1 for v in found.values())
    one = {k: v[0] for k, v in found.items() if len(v) == 1}
    used = {k: r.get('cuda_initialized', True) for k, r in one.items()}
    ooms = {k: r['num_ooms'] for k, r in one.items() if used[k]}
    recorded = recorded if recorded is not None else {k: recorded_ooms(cfg) for k, (_, cfg) in exp.items()}
    empty = {'cost_json': False, 'oom_arms': {}, 'completed_arms': {}}
    inside = {k: sum(recorded.get(k, empty)['oom_arms'].values()) + sum(recorded.get(k, empty)['completed_arms'].values()) for k in ooms}
    outside_arms = {k: (n or 0) - inside[k] for k, n in ooms.items() if (n or 0) != inside[k]}
    completed_with = {k: recorded[k]['completed_arms'] for k in ooms if recorded.get(k, empty)['completed_arms']}
    no_count = {k: sorted(a for a, n in recorded[k]['oom_arms'].items() if not n) for k in ooms if any(not n for n in recorded.get(k, empty)['oom_arms'].values())}
    known = bool(exp) and complete and not unexpected
    dev, total = [], None
    g = slurm / f'gpumem-{job}.csv'
    if g.exists():
        for row in csv.reader(g.read_text().splitlines()):
            if len(row) == 7 and row[2].strip().isdigit() and row[3].strip().isdigit():     # timestamp, uuid, used, total, SM clock, power, temperature
                dev.append(int(row[2])); total = int(row[3])
    retries = {k: r['num_alloc_retries'] for k, r in one.items() if used[k]}
    min_free = (total - max(dev)) / 1024 if dev else None
    return {'job': str(job), 'started_runners': sorted(exp), 'memory_state_known': known, 'clean': known and not outside_arms and not completed_with and not no_count,
            'records_complete': complete, 'missing_or_duplicate_records': sorted(k for k, v in found.items() if len(v) != 1), 'gpu_used': used, 'num_ooms': ooms,
            'recorded_increases': {k: recorded.get(k, empty) for k in ooms}, 'ooms_outside_arms': outside_arms, 'completed_arms_with_ooms': completed_with,
            'oom_arms_without_allocator_count': no_count, 'num_alloc_retries': retries,
            'peak_since_last_reset_reserved_gib': {k: one[k]['max_memory_reserved'] / GIB for k in ooms},
            'peak_since_last_reset_allocated_gib': {k: one[k]['max_memory_allocated'] / GIB for k in ooms},
            'alloc_conf': sorted({str(r.get('alloc_conf')) for r in recs}), 'unexpected_records': unexpected,
            'device_samples': len(dev), 'device_peak_used_gib': max(dev) / 1024 if dev else None, 'device_min_free_gib': min_free, 'device_total_gib': total / 1024 if total else None,
            'near_full': any((x or 0) > 0 for x in retries.values()) or (min_free is not None and min_free < 1),
            'rule': 'descriptive (E23 runs one process at a time): the memory state is known only if every started runner left one exit record and no record is unexpected; '
                    'num_ooms of a process that used the GPU is compared with the per-arm increases in its cost.json; OOMs outside every arm, completed arms with an increase, '
                    'out_of_memory arms without an increase and near_full (retries > 0 or minimum free < 1 GiB) must be flagged in the results; clean = known and none of the first three'}


if __name__ == '__main__':
    out = check(sys.argv[1]); dst = E / 'slurm/memory-check.json'; assert not dst.exists(); dst.write_text(json.dumps(out, indent=2) + '\n'); print(json.dumps(out, indent=1))
