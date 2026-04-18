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
