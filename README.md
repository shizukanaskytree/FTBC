# FTBC: Forward Temporal Bias Correction for Optimizing ANN-SNN Conversion

[[ECCV 2024](https://eccv2024.ecva.net/)] [[Paper](https://arxiv.org/abs/2403.18388)] [[Poster](./docs/ECCV_2024_poster.pdf)]

This repository contains our implementation of FTBC, along with several baseline methods related to ANN-SNN conversion. Below is an overview of our proposed method.

<p align="center">
    <img src="docs/figs/overview.png" alt="Project Overview" width="500"/>
</p>

### Running Experiments

To run the experiments for each baseline, navigate to the `run_scripts` folder located under the respective baseline directories. Instructions for running each experiment are provided within.

### Citation

If you find this repository useful, please consider citing our paper:

```bibtex
@inproceedings{wu2024ftbc,
  title={FTBC: Forward Temporal Bias Correction for Optimizing ANN-SNN Conversion},
  author={Wu, Xiaofeng and Bojkovic, Velibor and Gu, Bin and Suo, Kun and Zou, Kai},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```

## Neural Networks journal extension

The [journal extension](journal_extension/) contains the implementation, recorded
results, experimental settings and reproduction instructions for **Time-Ordered
Event Fitting for Forward Temporal Bias Correction in ANN-to-SNN Conversion**
(Xiaofeng Wu, Velibor Bojkovic and Kun Suo; manuscript prepared for Neural Networks).
The ECCV 2024 implementations and citation above remain the conference version.
Start with `bash journal_extension/reproduce.sh` to verify the release and
rebuild the primary-result summary on a CPU, without datasets or checkpoints.
See its README for the separate solver check and the requirements for full model evaluation.
