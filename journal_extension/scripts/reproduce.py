"""Verify the archived release and aggregate existing primary comparisons on CPU."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/tables'
OUT = ROOT / 'generated'


def require(condition, description):
    if not condition:
        raise ValueError(description)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    verified = 0
    for line in (ROOT / 'MANIFEST.sha256').read_text().splitlines():
        digest, relative = line.split('  ', 1)
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT), f'Path outside release: {relative}')
        require(sha(path) == digest, f'Checksum mismatch: {relative}')
        verified += 1
    provenance = json.loads((ROOT / 'release-provenance.json').read_text())
    for entry in provenance['source_files']:
        require(sha(ROOT / entry['release_path']) == entry['sha256'], entry['release_path'])

    tables = {}
    expected_counts = {'paired-comparisons.csv': 261, 'evaluation-settings.csv': 18,
                       'calibration-images.csv': 864, 'source-checkpoints.csv': 6,
                       'fitting-costs.csv': 24}
    for name, count in expected_counts.items():
        with (DATA / name).open(newline='') as f:
            tables[name] = list(csv.DictReader(f))
        require(len(tables[name]) == count, f'Unexpected row count: {name}')
    rows = tables['paired-comparisons.csv']
    unique = set()
    groups = defaultdict(list)
    for row in rows:
        identity = tuple(row[k] for k in ['dataset', 'model', 'T', 'comparison', 'draw'])
        require(identity not in unique, f'Duplicate comparison: {identity}')
        unique.add(identity)
        difference = float(row['candidate_top1_percent']) - float(row['control_top1_percent'])
        require(math.isclose(difference, float(row['difference_pp']), abs_tol=1e-9), f'Difference: {identity}')
        paired_difference = 100 * (int(row['candidate_only_correct']) - int(row['control_only_correct'])) / int(row['n'])
        require(math.isclose(difference, paired_difference, abs_tol=1e-9), f'Paired counts: {identity}')
        ce_lower = float(row['candidate_cross_entropy']) < float(row['control_cross_entropy'])
        joint = ce_lower and float(row['p_accuracy_gain_one_sided_unadjusted']) < .05
        require(ce_lower == (row['cross_entropy_lower'] == 'True'), f'CE outcome: {identity}')
        require(joint == (row['joint_improvement'] == 'True'), f'Joint outcome: {identity}')
        require((ROOT / 'results/records' / Path(row['source_record']).name).is_file(), f'Missing record: {identity}')
        if row['comparison'] == 'threshold_d':
            require(row['candidate'] == 'Ours-96', f'Primary candidate: {identity}')
            groups[(row['dataset'], row['model'], int(row['T']))].append(row)
    require(len(groups) == 18 and sum(map(len, groups.values())) == 54, 'Primary coverage')
    summary = []
    for (dataset, model, steps), group in sorted(groups.items()):
        require({r['draw'] for r in group} == {'C1', 'C2', 'C3'} and len(group) == 3, 'Draw coverage')
        require(len({r['control'] for r in group}) == 1, 'Primary control consistency')
        values = [float(r['candidate_top1_percent']) for r in group]
        summary.append({'dataset': dataset, 'model': model, 'T': steps,
                        'control': group[0]['control'], 'ours_mean_percent': mean(values),
                        'ours_sample_sd_pp': stdev(values),
                        'control_mean_percent': mean(float(r['control_top1_percent']) for r in group),
                        'mean_gain_pp': mean(float(r['difference_pp']) for r in group),
                        'joint_improvement_draws': sum(r['joint_improvement'] == 'True' for r in group)})
    OUT.mkdir(exist_ok=True)
    with (OUT / 'primary-summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    lines = ['| Dataset | Model | T | Control | Ours mean ± sample SD (%) | Control mean (%) | Gain (pp) | Joint draws |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in summary:
        lines.append(f"| {r['dataset']} | {r['model']} | {r['T']} | {r['control']} | {r['ours_mean_percent']:.2f} ± {r['ours_sample_sd_pp']:.2f} | {r['control_mean_percent']:.2f} | {r['mean_gain_pp']:.2f} | {r['joint_improvement_draws']}/3 |")
    (OUT / 'primary-summary.md').write_text('\n'.join(lines) + '\n')
    report = {'passed': True, 'manifest_files_verified': verified,
              'registered_sources_verified': len(provenance['source_files']),
              'data_rows': {k: len(v) for k, v in tables.items()}, 'primary_rows': 54,
              'primary_settings': len(summary), 'model_experiments_run': False,
              'bootstrap_or_prediction_statistics_recomputed': False}
    (OUT / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
