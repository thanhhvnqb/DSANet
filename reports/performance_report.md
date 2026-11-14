# DSANet and VadCLIP performance report

Date: 2026-11-08. Evidence: eight JSON logs in `runs/`, covering four completed 10-epoch runs, one seed (234) per method/dataset. No new experiments were run. All scores below are percentages; differences are percentage points (pp). Higher is better.

## Main comparison

The first four rows are the fixed paper and reproduction references for future experiments. UCF uses frame ROC-AUC. XD uses the checkpoint-selection AP: DSANet binary C-branch and VadCLIP alignment A-branch. These are method-specific headline scores, not identical-branch measurements.

| Method / source | UCF AUC | XD AP | UCF average mAP | XD average mAP |
|---|---:|---:|---:|---:|
| VadCLIP — original paper | 88.02 | 84.51 | 6.68 | 24.70 |
| DSANet — original paper | 89.44 | 86.95 | 13.01 | 28.87 |
| VadCLIP — reproduced, seed 234 | 87.80 | 84.74 | 7.66 | 23.28 |
| DSANet — reproduced, seed 234 | 89.09 | 84.52 | 10.71 | 24.02 |

DSANet exceeds reproduced VadCLIP by **1.29 pp UCF AUC**, but trails it by **0.22 pp XD AP**. Its reproduced UCF AUC is 0.35 pp below its own paper and 1.07 pp above the VadCLIP paper. Its XD AP is 2.43 pp below its own paper and only 0.01 pp above the VadCLIP paper. Thus, the paper's XD advantage is not reproduced here.

VadCLIP is 0.22 pp below its own paper on UCF and 0.23 pp above it on XD; relative to the DSANet paper it is lower by 1.64 pp and 2.21 pp, respectively. DSANet improves average mAP over reproduced VadCLIP by 3.05 pp on UCF and 0.74 pp on XD, while remaining below its own paper by 2.30 pp and 4.85 pp. Relative to the VadCLIP paper, DSANet average mAP is +4.03 pp on UCF and -0.68 pp on XD. VadCLIP average mAP is +0.98/-1.42 pp versus its own paper and -5.35/-5.59 pp versus the DSANet paper (UCF/XD).

## Fine-grained localization

Average mAP is the arithmetic mean across IoU 0.1–0.5. Paper rows are from each method's original paper, Tables 3–4.

### UCF

| Method / source | mAP@0.1 | @0.2 | @0.3 | @0.4 | @0.5 | Average |
|---|---:|---:|---:|---:|---:|---:|
| VadCLIP — paper | 11.72 | 7.83 | 6.40 | 4.53 | 2.93 | 6.68 |
| DSANet — paper | 21.39 | 14.96 | 11.74 | 8.98 | 8.00 | 13.01 |
| VadCLIP — reproduced | 13.54 | 9.95 | 5.98 | 4.93 | 3.90 | 7.66 |
| DSANet — reproduced | 18.53 | 13.34 | 9.80 | 7.17 | 4.72 | 10.71 |

### XD

| Method / source | mAP@0.1 | @0.2 | @0.3 | @0.4 | @0.5 | Average |
|---|---:|---:|---:|---:|---:|---:|
| VadCLIP — paper | 37.03 | 30.84 | 23.38 | 17.90 | 14.31 | 24.70 |
| DSANet — paper | 40.93 | 34.63 | 28.21 | 22.70 | 17.89 | 28.87 |
| VadCLIP — reproduced | 35.50 | 29.57 | 21.96 | 16.69 | 12.68 | 23.28 |
| DSANet — reproduced | 35.20 | 28.89 | 23.81 | 18.49 | 13.71 | 24.02 |

## Training selection and evaluation consistency

| Run | Best epoch (1-based) | Validation events | Selection metric | Best validation | Standalone test |
|---|---:|---:|---|---:|---:|
| VadCLIP / ucf | 6 | 120 | auc | 87.8022 | 87.8022 |
| VadCLIP / xd | 10 | 10 | alignment_ap | 84.7353 | 84.7353 |
| DSANet / ucf | 9 | 120 | frame_level_roc_auc | 89.0936 | 89.0936 |
| DSANet / xd | 1 | 10 | frame_level_average_precision | 84.5190 | 84.5190 |

All four primary test scores match their selected validation scores. UCF validates within epochs (120 events); XD validates at epoch end (10 events). All runs use the test split for checkpoint selection, so standalone evaluation is not an independent held-out estimate. Only one seed is available; uncertainty and statistical significance cannot be estimated.

DSANet UCF fine-grained mAP differs between selection-time validation (10.5073) and standalone testing (10.7138), although binary AUC matches. The current source uses lowercase training descriptions but capitalized standalone-test descriptions, affecting text alignment. The test entry point has now been aligned with training for future evaluations. Historical tables retain the saved standalone-test values; no corrected test result is claimed. XD metrics match exactly. The saved training algorithms also reload the best weights after every epoch; these logs should not be interpreted as ordinary uninterrupted optimization trajectories.

## Branch diagnostics

These values are reproduced only; no corresponding original-paper branch measurements are claimed. Abnormal-only metrics use the binary branch in these logs.

| Method / dataset | C AUC | C AP | A AUC | A AP | Abnormal-only AUC | Abnormal-only AP |
|---|---:|---:|---:|---:|---:|---:|
| VadCLIP / ucf | 87.80 | 33.49 | 85.94 | 27.40 | 70.14 | Unavailable |
| VadCLIP / xd | 93.73 | 79.39 | 94.75 | 84.74 | 79.84 | 79.57 |
| DSANet / ucf | 89.09 | 34.91 | 88.56 | 31.73 | 73.62 | 35.62 |
| DSANet / xd | 94.59 | 84.52 | 94.13 | 83.17 | 83.29 | 84.87 |

For identical binary branches on XD, DSANet AP exceeds VadCLIP by 5.13 pp (84.52 vs 79.39). For alignment branches, DSANet is lower by 1.56 pp (83.17 vs 84.74). This explains why branch choice materially affects the comparison. Fine-grained score construction also differs between methods; it should not be treated as a controlled module ablation.

## Sources and artifact provenance

- [VadCLIP original paper](https://arxiv.org/html/2308.11681), Tables 1–4; [AAAI publication](https://ojs.aaai.org/index.php/AAAI/article/view/28423).
- [DSANet original paper](https://arxiv.org/html/2511.10334v1), Tables 1–4.
- Historical logs remain at their original locations below. Embedded paths describe the original execution and may differ from today's artifact location. DSANet model files are still under `model/`; VadCLIP weights are stored beside its logs. The shared historical `model/checkpoint.pth` cannot establish provenance for every run, so it has not been duplicated or relabeled as four verified best checkpoints.

- VadCLIP / ucf: [test](../runs/VadCLIP/ucf/seed_234/ucf_test_20261108_015341_098871.json), [train](../runs/VadCLIP/ucf/seed_234/ucf_train_20261107_173343_096476.json).
- VadCLIP / xd: [test](../runs/VadCLIP/xd/seed_234/xd_test_20261108_015021_508063.json), [train](../runs/VadCLIP/xd/seed_234/xd_train_20261107_185134_651447.json).
- DSANet / ucf: [test](../runs/DSANet/ucf/seed_234/test_20261108T015458.174952Z.json), [train](../runs/DSANet/ucf/seed_234/train_20261107T150831.801630Z.json).
- DSANet / xd: [test](../runs/DSANet/xd/seed_234/test_20261108T093342.529591Z.json), [train](../runs/DSANet/xd/seed_234/train_20261107T163449.930973Z.json).
