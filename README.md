<div align="center">

<h2 align="center">
Learning to Tell Apart: Weakly Supervised Video Anomaly Detection via Disentangled Semantic Alignment (AAAI 2026)
</h2>

<p align="center">
  <a href="https://ojs.aaai.org/index.php/AAAI/article/view/38191">
    <img src="https://img.shields.io/badge/Paper-AAAI%202026-000080.svg" alt="AAAI Official Paper">
  </a>
  <a href="https://arxiv.org/abs/2511.10334">
    <img src="https://img.shields.io/badge/Paper-arXiv%3A2511.10334-b31b1b.svg" alt="arXiv Paper">
  </a>
  <a href="https://drive.google.com/drive/folders/1PqvaNm_s-fOOrnJRqrG50zV2R2UqRwHK?usp=sharing">
    <img src="https://img.shields.io/badge/Weights-Google%20Drive-blue.svg" alt="Weights">
  </a>
</p>



</div>

> [!NOTE]
> Our paper has been accepted by **AAAI 2026**! This is the official PyTorch implementation for **DSANet**.

## 📖 Abstract

Recent advancements in weakly-supervised video anomaly detection have achieved remarkable performance by applying the multiple instance learning paradigm based on multimodal foundation models such as CLIP to highlight anomalous instances and classify categories.
However, their objectives may tend to detect the most salient response segments, while neglecting to mine diverse normal patterns separated from anomalies, and are prone to category confusion due to similar appearance, leading to unsatisfactory fine-grained classification results.
Therefore, we propose a novel Disentangled Semantic Alignment Network (DSANet) to explicitly separate abnormal and normal features from coarse-grained and fine-grained aspects, enhancing the distinguishability.
Specifically, at the coarse-grained level, we introduce a self-guided normality modeling branch that reconstructs input video features under the guidance of learned normal prototypes, encouraging the model to exploit normality cues inherent in the video, thereby improving the temporal separation of normal patterns and anomalous events.
At the fine-grained level, we present a decoupled contrastive semantic alignment mechanism, which first temporally decomposes each video into event-centric and background-centric components using frame-level anomaly scores and then applies visual-language contrastive learning to enhance class-discriminative representations.
Comprehensive experiments on two standard benchmarks, namely XD-Violence and UCF-Crime, demonstrate that DSANet outperforms existing state-of-the-art methods.
f
f
dsa
fsda
fsa

<div align="center">
  <img src="assets/framework.png" width="100%" alt="Framework of DSANet"/>
</div>

## 🛠️ Environment

The code is developed and tested on a single **NVIDIA RTX 4090** GPU.

To set up the environment, please run:

```bash
conda env create -f environment.yml
conda activate dsanet
```

## 📂 Data Preparation

### Features
We utilize the pre-extracted CLIP features provided by **VadCLIP**. Please download the features for UCF-Crime and XD-Violence datasets from their official repository:

- **VadCLIP Repository:** [GitHub - nwpu-zxr/VadCLIP](https://github.com/nwpu-zxr/VadCLIP)

### Directory Structure
After downloading, please organize the features and update the paths in the csv files located in the `list/` directory:

1. Open `list/ucf_CLIP_rgb.csv` (or similar .csv files) and replace the feature paths with your local directory paths.
2. Ensure the annotations match the feature files.

## 🚀 Training

To train the model on a specific dataset, ensure you have updated the feature paths in the csv files first.

**UCF-Crime:**
```bash
python src/ucf_train.py
```

**XD-Violence:**
```bash
python src/xd_train.py
```

Training checkpoints are loaded with `weights_only=True`. Newly saved
checkpoints store the selection metric as a Python float. Older checkpoints
containing a NumPy scalar may still fail restricted loading; converting them
requires loading with `weights_only=False`, which can execute arbitrary code.
Only do so for a checkpoint from a trusted source.

New training and test outputs use
`runs/<hypothesis>/<variant>/<dataset>/seed_<seed>/` (defaults: `baseline`,
`DSANet`, seed `234`). Each directory contains unique timestamped training/test
JSON logs, `best_checkpoint.pth` (training state), and `best_model.pth`
(inference weights). UCF also saves `last_model.pth`. Best weights are saved
immediately whenever the selection metric improves. Existing model artifacts
are protected from accidental fresh-run overwrites; choose a new variant for a
repeat run. Explicit output paths must remain inside the run directory.
Standalone testing accepts `--model-path` to evaluate external weights.
`--use-checkpoint True` explicitly resumes from the run checkpoint using the
existing training resume behavior; exact scheduler/RNG continuation is not
implemented. Logs retain configuration, environment, Git metadata, training
losses, validation events, and selected-checkpoint metrics. Selection uses the
test split and must be disclosed in reports.

Run from the repository root (commands are provided for user execution):

```bash
rtk proxy .venv/bin/python src/ucf_train.py --hypothesis baseline --variant DSANet --seed 234
rtk proxy .venv/bin/python src/ucf_test.py --hypothesis baseline --variant DSANet --seed 234
rtk proxy .venv/bin/python src/xd_train.py --hypothesis baseline --variant DSANet --seed 234
rtk proxy .venv/bin/python src/xd_test.py --hypothesis baseline --variant DSANet --seed 234
```

Historical artifacts retain their previous paths. The implementation in this
repository trains DSANet; naming a variant `VadCLIP` does not change the model.
VadCLIP logs are imported baseline evidence, and its separate implementation
is outside this repository's training entry points.

See [the performance report](reports/performance_report.md) for original-paper
and reproduced results, branch definitions, and evaluation limitations.

## 🧪 Testing

We provide the pre-trained model weights to reproduce our results.

1. **Download Weights:** Download the `.pth` files from [Google Drive](https://drive.google.com/drive/folders/1PqvaNm_s-fOOrnJRqrG50zV2R2UqRwHK?usp=sharing).
2. **Placement:** Place downloaded weights in a chosen location and pass `--model-path model/model_ucf.pth` or `--model-path model/model_xd.pth`. Without this option, testing loads the selected run directory’s `best_model.pth`.

**Evaluate on UCF-Crime:**
```bash
python src/ucf_test.py
```

**Evaluate on XD-Violence:**
```bash
python src/xd_test.py
```

Validation prints the checkpoint-selection metric (UCF frame-level ROC-AUC or
XD frame-level AP), fine-grained mAP at IoU 0.1–0.5, abnormal-video-only
ROC-AUC and AP, and branch diagnostics. In this implementation, the C-branch
is the binary anomaly detector (`sigmoid(logits1)`), and the A-branch is the
vision-language alignment score (`1 - softmax(logits2)[normal]`). The
checkpoint metric and abnormal-only metrics use the C-branch; fine-grained
mAP retains the existing temperature-scaled hierarchical probabilities.
Scores are repeated 16 times to map snippets to frames. Model selection uses
the test set, so these results are test-selected, not an unbiased held-out test
estimate.

## 📊 Results

| Method | UCF-Crime (AUC/%) | XD-Violence (AP/%) |
| :--- | :---: | :---: |
| **DSANet (Ours)** | **89.44** | **86.95** |

> Note: Please refer to our paper for the detailed performance comparison.

## 🙏 Acknowledgement

We thank the authors of the following repositories for their valuable work and codebases:

* [VadCLIP](https://github.com/nwpu-zxr/VadCLIP)
* [INP-Former](https://github.com/luow23/INP-Former)
* [AA-CLIP](https://github.com/Mwxinnn/AA-CLIP)

## 📜 Citation

If you find this repository useful for your research, please consider citing our paper:

```bibtex
@inproceedings{yin2026learning,
  title={Learning to Tell Apart: Weakly Supervised Video Anomaly Detection via Disentangled Semantic Alignment},
  author={Yin, Wenti and Zhang, Huaxin and Wang, Xiang and Lu, Yuqing and Zhang, Yicheng and Gong, Bingquan and Zuo, Jialong and Yu, Li and Gao, Changxin and Sang, Nong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={14},
  pages={12027--12035},
  year={2026}
}
