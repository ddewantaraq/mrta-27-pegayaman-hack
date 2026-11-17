# Detector fine-tune v3 — boundaries (frozen before training/smoke)

## Training (external only, legal)

- Data: skaczylo/Football-Image-Dataset (CC BY 4.0), 612 Full HD
  modern match frames, YOLO format, author split 512 train / 100 val.
  No BAR-ATH data, labels, or tuning anywhere in training/selection.
- Recipe (dataset author's data.yaml recommendation, frozen):
  YOLO11n pretrained, imgsz 1280, epochs 100, default augmentation.
- Selection: ball-class results on the dataset's own val split only.

## Smoke (unchanged from v2)

- Winning fine-tuned weights frozen by sha256 before any BAR-ATH run.
- Identical segment 11185–13184, ROI gate 5 m, kill = median
  disagreement > 2 m (gated) or zero gated detections.
- Aggregates only; no changes after output is seen; all v1/v2
  boundaries remain in force.
