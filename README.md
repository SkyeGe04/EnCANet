# EnCANet: A Network with Entropy-Guided Saliency and Cross-Spatial Attention for Remote Sensing Change Detection

EnCANet is a concise, production‑ready pipeline for binary change detection on bitemporal remote‑sensing images. It ships a complete workflow: data preparation → training → evaluation/inference → export of predictions.

Key highlights:
- Fixed modules by design: EnFoCSA (cross‑spatial attention) and ESP (entropy‑based selection) are always enabled
- Two‑stage fusion (temporal + spatial) implemented via Uni‑FiRE blocks
- EfficientNetV2‑S encoder with ImageNet normalization
- Simple, reproducible entry points (`main.py` for train→test, `inference.py` for test‑only)

---

## Table of Contents
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Configuration](#configuration)
- [Train](#train)
- [Inference / Evaluation](#inference--evaluation)
- [Outputs](#outputs)
- [Requirements](#requirements)
- [FAQ](#faq)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Repository Structure

```
.
├── main.py                  # Train then test
├── train.py                 # Training loop and validation
├── inference.py             # Test / export predictions
├── models/
│   ├── EnCANet.py           # Model class and factory (getEnCANet)
│   ├── efficientnet.py      # EfficientNetV2‑S encoder wrapper
│   ├── fusion_modules.py    # TwoStageUniFiREFusion / UniFiRE blocks
│   └── rccd_layers.py       # EnFoCSAModule (cross‑spatial attention)
├── utils/
│   ├── data_loading.py      # Dataset / transforms (Albumentations)
│   ├── dataset_process.py   # Tools for cropping/splitting/mean-std
│   ├── losses.py            # loss() = Dice + BCE
│   ├── path_hyperparameter.py # Global hyperparameters (ph)
│   └── utils.py             # train_val loop, checkpoint, metrics
└── scripts...               # Optional helpers (convert, split, etc.)
```

## Quick Start

1) Create environment (Python 3.10+; install PyTorch matching your CUDA):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # example
pip install albumentations torchmetrics pillow numpy tqdm wandb prefetch-generator
```

2) Organize your dataset as below (file names must match across t1/t2/label):

```
./{dataset_name}/
  ├── train/  val/  test/
  │   ├── t1/        # timestamp 1 RGB images
  │   ├── t2/        # timestamp 2 RGB images
  │   └── label/     # binary masks (0/255 or 0/1)
```

3) Train then test (default config in `utils/path_hyperparameter.py`):

```bash
python main.py --dataset_name LEVIR-CD
```

4) Test only with an existing checkpoint:

```bash
python main.py --dataset_name LEVIR-CD --skip_train --model_path ./LEVIR-CD_best_f1score_model/xxx.pth
```

5) Test only via `inference.py` (if `ph.load` is already set):

```bash
python inference.py
```

## Data Preparation

If you start from a flat folder `t1/ t2/ label/`, use the provided utilities:

```bash
# Split into train/val/test
python split_dataset.py

# Optional: cropping/tiling
python preprocess_data.py
```

## Configuration

Global hyperparameters live in `utils/path_hyperparameter.py` and are referenced via `from utils.path_hyperparameter import ph`.

Commonly adjusted fields (see file for full list):
- `dataset_name`: dataset root folder name
- `epochs`, `batch_size`, `learning_rate`, `weight_decay`, `amp`, `max_norm`
- `evaluate_epoch`, `save_checkpoint`, `save_interval`
- `patch_size`, `cropping_method`, `pre_size`, `overlap_size`
- `backbone_name` (default: `efficientnetv2_s_22k`) and `pretrained`
- `temporal_fusion_method`, `spatial_fusion_method` (default: `channel_concat`)

Notes:
- Image normalization is fixed to ImageNet mean/std.
- EnFoCSA and ESP are always enabled inside the model; no switches required.

## Train

```bash
python main.py --dataset_name YOUR_DATASET \
  --epochs 200 --batch_size 16 --learning_rate 1e-4
```

During training:
- Metrics are computed with torchmetrics (Accuracy/Precision/Recall/F1/IoU)
- Checkpoints and best models are saved under `./{dataset}_checkpoint/`, `./{dataset}_best_f1score_model/`, `./{dataset}_best_loss_model/`
- Weights & Biases logging is enabled (defaults to offline mode in code); set your API key or keep offline

## Inference / Evaluation

```bash
python main.py --dataset_name YOUR_DATASET --skip_train --model_path /path/to/model.pth
```

or

```bash
python inference.py
```

The script computes metrics over the test set and exports binary predictions to:

```
./{dataset_name}/pred/
```

Predictions adopt the original file extension (falls back to `.png` if unknown).

## Outputs

- `pred/`: binary masks (0/255)
- `*_checkpoint/`: periodic checkpoints
- `*_best_f1score_model/` and `*_best_loss_model/`: best models selected during validation

## Requirements

- Python 3.10+
- PyTorch / CUDA (match your system)
- Python packages: `torch`, `torchvision`, `albumentations`, `torchmetrics`, `numpy`, `pillow`, `tqdm`, `wandb`, `prefetch-generator`

## FAQ

Q1: My dataset masks are 0/1. Is that OK?

Yes. The code accepts 0/1 or 0/255; masks are converted accordingly.

Q2: Where are predictions saved?

`./{dataset}/pred/` as single‑channel binary images.

Q3: How do I resume from a checkpoint?

Set `ph.load` to a checkpoint path or pass `--model_path` in `main.py --skip_train` mode.

## Acknowledgements

This repository builds upon common practices in remote‑sensing change detection and uses open‑source libraries, notably PyTorch, Albumentations and TorchMetrics.

## License

This project is released under the MIT License. See `LICENSE` for details.



