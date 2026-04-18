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

Latest verified run used configuration from `waste_detection/constant/training_pipeline/__init__.py`:

- `MODEL_TRAINER_PRETRAINED_WEIGHT_NAME`: `yolov5s.pt`
- `MODEL_TRAINER_NO_EPOCHS`: `500`
- `MODEL_TRAINER_BATCH_SIZE`: `64`
- Image size: `416`
- Number of classes: `13`

Final validation summary (best model after 500 epochs):

- Precision (P): `0.983`
- Recall (R): `0.957`
- mAP@0.5: `0.979`
- mAP@0.5:0.95: `0.928`

Interpretation:

- Model quality is already strong and stable: precision and recall are both high with balanced detection behavior.
- The gap between mAP@0.5 and mAP@0.5:0.95 is relatively small, indicating good localization quality, not only coarse detection.
- Training in late epochs (around 480-500) is already plateauing, so additional epochs will likely give only marginal gains.

Class-level highlights (mAP@0.5 / mAP@0.5:0.95):

- Strongest classes: `drinkcan` (0.995 / 0.970), `tissueroll` (0.995 / 0.959), `plasticbottle` (0.995 / 0.938), `drinkpack` (0.994 / 0.952), `lettuce` (0.995 / 0.925).
- Good but still improvable classes: `plasticcontainer` (0.972 / 0.879), `sweetpotato` (0.962 / 0.891), `plasticbag` (0.939 / 0.905), `chilli` (0.955 / 0.900).

Recommended next steps:

1. Save this run as baseline production candidate because overall metrics are already high.
2. Prioritize targeted data improvement on `plasticcontainer`, `sweetpotato`, and `plasticbag` (more hard examples and label review).
3. Consider enabling early stopping in future runs to reduce compute when metrics plateau.
4. Evaluate on a true hold-out test set (or real camera data) before deployment to confirm generalization.
