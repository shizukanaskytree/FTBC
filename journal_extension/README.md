# Neural Networks journal extension

**Time-Ordered Event Fitting for Forward Temporal Bias Correction in ANN-to-SNN Conversion**

Xiaofeng Wu, Velibor Bojkovic and Kun Suo. Manuscript prepared for *Neural Networks*, 15 October 2026.

This directory separates the journal study from the ECCV 2024 implementation in
`../QCFS/`, `../RTS/` and `../SNNC/`. The journal study fits timestep-specific
corrections sequentially on the deployed integrate-and-fire trajectory. Its
FP32 implementation uses bounded bisection; the real-arithmetic theoretical
construction and the numerical implementation have different guarantees.

## Contents

| Path | Content |
| --- | --- |
| `frozen_source/` | All 242 Python files registered by the study's frozen implementation manifest, with their original bytes and relative stage layout. |
| `results/tables/` | 261 comparisons (including 54 primary rows), 18 evaluation settings, 864 calibration-image identifiers, six checkpoint identities, 24 fitting-cost records, software environments and source identities. |
| `results/records/` | The ten original acceptance/analysis JSON records underlying the comparison CSV. |
| `results/figures/` | The existing numerical and result figures from the manuscript; no new model runs were used to make this release. |
| `configs/original/` | The ten frozen main-job configurations, preserving the historical paths, seeds, controls and numerical settings. |
| `scripts/` and `reproduce.sh` | CPU verification and result aggregation; an optional test of the original numerical solver. |
| `release-provenance.json` | Original-to-release file mapping and the registered source hashes. |

## Rebuild the reported result summary

Python 3.10 or later is sufficient; no third-party packages, dataset downloads or
model checkpoints are needed for this command:

```bash
bash journal_extension/reproduce.sh
```

It verifies the release checksums and source identities, checks the comparison
records, and writes `generated/primary-summary.csv`, `generated/primary-summary.md`
and `generated/verification.json`. It also recomputes the control-sensitivity
summary against the included original records. These are calculations from the
published records, **not a rerun of model inference or training**. The bootstrap
intervals and p-values remain the stored values; individual predictions are not
included here, so this command does not recompute those statistics from images.

Primary rows use `comparison == threshold_d`. Primary comparisons also occur
among secondary rows; do not average all 261 rows as independent observations.
The primary FTBC control was selected by aggregate evaluation accuracy across
three calibration draws. Tests are unadjusted for this selection and multiplicity.
Draws share source networks and evaluation images. See `results/tables/README.md`
for the statistical definitions and recorded environment limitations.

## Check the original solver on CPU

Install the dependencies in a suitable environment, or use an existing environment
that satisfies `requirements.txt`. The frozen model import chain also loads
Matplotlib and SymPy, even though this check does not render plots. Their original
version numbers were not recorded.

```bash
python -m pip install -r journal_extension/requirements.txt
PYTHON=/path/to/python bash journal_extension/reproduce.sh --solver-check
```

This runs the frozen `c1-pre/e14/source/check_step_solver.py` on synthetic CPU
tensors. It checks bitwise agreement with the original IF forward recurrence,
per-step means and per-step monotonicity. It initializes no CUDA context and
does not read any benchmark images. Its output is a numerical implementation
check, not validation of the paper's classification accuracy.

## Full model evaluation

The exact evaluation entry points are
`frozen_source/locked-eval-v2/source/l1_imagenet_runner.py` and
`frozen_source/locked-eval-v2/source/l1_cifar_runner.py`. The main fitting solver is
`frozen_source/c1-pre/e14/source/step_solver.py`. The original runners retain
their dependency chain and configuration checks.

**The archived GPU workflow is not a portable, one-command experiment release.**
It refers to the original workspace, data manifests, deployment artifacts and
checkpoint files. The ten configurations are reference records, not directly
runnable examples on a new machine. A full rerun requires obtaining the exact
weights and deployment artifacts, recreating the recorded data selections,
mapping the original paths, and validating the adapted workflow. No full GPU
rerun or validation of such a port was performed for this release. Historical
job-submission scripts are included for provenance; do not run them as setup commands.

CIFAR-10/100 must be obtained from the [original provider](https://www.cs.toronto.edu/~kriz/cifar.html).
ImageNet must be obtained under its [access terms](https://www.image-net.org/download.php).
No dataset images or model-weight files are redistributed. Exact weights and
the additional deployment artifacts can be requested from Xiaofeng Wu at
xiaofengwu@cityu.edu.mo; `results/tables/source-checkpoints.csv` identifies the
six source states. Check their SHA-256 identities rather than substituting an
older ECCV checkpoint with the same architecture.

The recorded GPU runs used PyTorch 2.5.0 with CUDA 12.4. CIFAR records also list
torchvision 0.20.0, NumPy 2.2.6 and Pillow 12.0.0. ImageNet did not record all
those package versions. `requirements.txt` distinguishes the recorded core
from auxiliary dependencies whose historical versions are unknown.

On the authors' server, any new GPU evaluation must use a persistent Slurm
allocation through `gpurun`. This release does not start GPU jobs.

## Attribution and versioning

The original ECCV citation and all baseline directories remain unchanged.
Source comments and attribution in the archived files are retained. This
addition does not assign a new license to third-party code or datasets.
Use a fixed Git commit when citing this release. A Zenodo DOI has not been
created; a GitHub URL must not be represented as a Zenodo deposit.

The journal manuscript is not represented here as an accepted or published paper.
For the published FTBC foundation, cite the ECCV 2024 paper listed in the root README.
