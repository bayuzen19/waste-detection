# Waste Detection

A computer vision project for waste object detection using YOLOv5 with a modular Python training pipeline.

## Project Structure

- `data/`: Dataset configuration and train/validation data.
- `research/`: Experiment notebooks and YOLOv5 workspace.
- `waste_detection/`: Core package for ingestion, validation, training, and utilities.
- `app.py`: Pipeline entry point.

## Quick Start

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the training pipeline:

```bash
python app.py
```

## Notes

- YOLOv5 experiment workflows are kept under `research/yolov5`.
- Generated artifacts and logs are excluded from Git via `.gitignore`.
