# Waste Detection

A computer vision project for waste object detection using YOLOv5 with a modular Python training pipeline.

## Project Structure

- `data/`: Dataset configuration and train/validation data.
- `research/`: Experiment notebooks and YOLOv5 workspace.
- `waste_detection/`: Core package for ingestion, validation, training, and utilities.
- `app.py`: Pipeline entry point.

## Quick Start

1. Create and activate the conda environment.

```bash
conda create -n waste python=3.7 -y
conda activate waste
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Install project package in editable mode.

```bash
pip install -e .
```

4. Run the training pipeline.

```bash
python app.py
```

## Troubleshooting

- If `python app.py` fails with missing module errors, check interpreter mismatch.
- Verify active Python:

```bash
which python
python --version
```

- If needed, run with the environment interpreter explicitly:

```bash
/home/bayuzen/anaconda3/envs/waste/bin/python app.py
```

## Notes

- YOLOv5 experiment workflows are kept under `research/yolov5`.
- Generated artifacts, logs, and experiment outputs are excluded from Git via `.gitignore`.

## YOLO Training Result Analysis

Latest verified run used YOLOv5s with the following setup:

- Epochs: 1
- Batch size: 16
- Image size: 416
- Classes: 13

Validation summary (best model):

- Precision (P): 0.00708
- Recall (R): 0.765
- mAP@0.5: 0.0481
- mAP@0.5:0.95: 0.0171

Interpretation:

- Recall is already high for an early run, which means many true objects are being detected.
- Precision is still very low, indicating many false positives.
- mAP values are low because one epoch is not enough for stable localization and class separation across 13 classes.

Class-level snapshot:

- Stronger early classes by mAP@0.5: `tissueroll` (0.134), `banana` (0.116), `foodcan` (0.0783).
- Weaker early classes include `paperbag`, `plasticbag`, and `drinkcan`, which suggests class confusion and limited learning time.

Recommended next training steps:

1. Increase epochs to at least 50-100 and monitor overfitting.
2. Keep validation monitoring on mAP@0.5:0.95, not only recall.
3. Review label quality for low-performing classes and rebalance data if class counts are skewed.
4. Optionally tune augmentation and learning rate schedule after baseline convergence.
