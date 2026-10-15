# Experimental settings and data

This directory accompanies Section 5 and Supplementary Section S3. `paired-comparisons.csv` is Supplementary Data 1. It exports existing frozen records; the export does not train, fit, run inference, select configurations, or change historical statistical decisions.

The CSVs preserve complete filenames and values without PDF line wrapping. The manuscript's separate `FTBC_Supplementary_Data.zip` contains the same numerical data. For this public repository, run `bash journal_extension/reproduce.sh` from the repository root to verify the files and rebuild the primary summary.

| File | Contents |
| --- | --- |
| `paired-comparisons.csv` | 261 per-draw comparisons: primary and secondary, including 54 primary rows. Accuracy, paired counts, both one-sided p-values, bootstrap intervals and seeds, both cross-entropies, and separate accuracy/CE outcomes. |
| `evaluation-settings.csv` | 18 dataset/model/timestep settings, source identity, evaluation and capture batches, execution seed, selected FTBC settings and primary control. |
| `calibration-images.csv` | The ordered 96 images for each of three draws on each dataset (864 rows). The first 32 also identify Ours-32. |
| `source-checkpoints.csv` | Six source checkpoint filenames, quantization levels, state-dictionary formats/entries and registered file digests. An empty key means that the loaded object itself is the state dictionary; only the CIFAR-100 ResNet-20 file uses the nested `model` entry. These refer to this study's source states, not the earlier conference models. |
| `recorded-environments.json` | Software versions and GPU types actually recorded for each evaluation job. Missing ImageNet package versions are not inferred from today's environment. |
| `implementation-versions.json` | The frozen evaluation revision and its registered Python-source identities. |
| `fitting-costs.csv` | 24 measurements: two repetitions for two methods, two ImageNet models and three timesteps, with measured auxiliary time components and memory. |
| `export-summary.json` | Counts and source-record locations for the export. |
| `control-sensitivity.json` | Original recorded summary of the two FTBC controls: seven CIFAR settings have different joint outcomes; the selected control gives fewer qualifying draws in six and more in one. |
| `../../scripts/summarize_control_sensitivity.py` | Regenerates this summary as JSON and LaTeX in `../../generated/`. Its `--verify-frozen` option checks the included JSON records in `../records/`. |

The comparison symbol `+` requires an exact one-sided McNemar p-value below 0.05 for an accuracy gain **and** lower mean cross-entropy. `-` requires the opposite one-sided p-value below 0.05. `0` means neither of these criteria is met; it includes significant accuracy gains that do not reduce cross-entropy. `frozen_record_label` preserves the historical wording verbatim. Use `accuracy_test`, `cross_entropy_lower` and `joint_improvement` to distinguish the conditions. All p-values are unadjusted; two-sided 95% intervals use 10,000 paired bootstrap resamples. Values are exported at the stored precision; extremely small floating-point p-values may be stored as zero.

The primary control is the FTBC configuration with the most correct predictions aggregated over the three draws on the evaluation images. Neither the p-values nor the paired intervals adjust for this outcome-based selection or for multiplicity. Primary and secondary comparisons are descriptive at the per-comparison level. The `source_record` paths preserve the original workspace locations; the ten corresponding JSON files are included by basename in `../records/`.

Primary comparisons also appear among the secondary rows. When the tuned FTBC configuration equals FTBC-96, there is no second run; the identical comparison is not duplicated. The row count is therefore 261 rather than 270. The data share source checkpoints and evaluation images across calibration draws, and do not represent independently retrained networks.

Calibration lists use zero-based CIFAR training indices (official batch order) and ImageNet paths relative to the training directory. Each listed calibration image was checked against the frozen exclusions. The sampling seeds identify the PCG64 permutation, while the lists fix the actual selected images after exclusions. No dataset images are redistributed here.

The cost `fitting_seconds` subtracts the separately measured data, target, full-network replay, write, digest and hash components where applicable. It retains uninstrumented overhead and, for Ours-96, the per-layer replay assertions. Memory is peak allocated memory above the resident baseline, not total GPU memory. The two repetitions used the same images and did not restart the process. Development search/scoring cost is excluded.

## Provenance

These CSVs were exported from the frozen research records by the manuscript's `export_experiment_artifacts.py`. That original workspace export and the manuscript's LaTeX cost table belong to the submission sources; they are not standalone commands or files in this directory. The portable entry point for the public release is `../../reproduce.sh`.

## Availability

The registered implementation sources are included in `../../frozen_source/`; the mapping from original paths and hashes is in `../../release-provenance.json`. The extension README distinguishes the portable CPU commands from the archived GPU workflow. Dataset images, exact checkpoint weights and the original deployment artifacts are not included. The exact weights and deployment artifacts are available from Xiaofeng Wu (xiaofengwu@cityu.edu.mo) upon reasonable request. Checkpoint filenames and identities are in `source-checkpoints.csv`.

The ImageNet runs recorded PyTorch, CUDA and cuDNN, but did not separately record torchvision, NumPy and Pillow versions. The CIFAR run records contain those additional versions. These omissions are retained explicitly rather than reconstructed from an unrelated later environment.
