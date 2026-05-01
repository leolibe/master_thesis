# Master Thesis - Leo Liberkowski: Road Segmentation from Norwegian Orthophotos

This project develops a deep learning pipeline for road segmentation using aerial orthophotos and vector road data.

## Objective

Generate binary road masks from geospatial data and train a U-Net model to segment roads from orthophotos.

## Data

- Orthophotos: GeoTIFF, 30 cm/pixel
- CRS: EPSG:25833
- Vector roads: GML
- Masks: binary PNG masks

Raw data is not included in this repository due to size and licensing constraints.

## Pipeline

1. Load orthophotos and GML road geometries
2. Convert geometries to EPSG:25833
3. Rasterize roads into binary masks
4. Generate aligned image/mask patches
5. Filter empty patches
6. Split dataset by tile to avoid spatial leakage
7. Train U-Net segmentation model
8. Evaluate using Dice, IoU, Precision, Recall

## Results

Example test-set metrics:

| Metric | Value |
|---|---|
| Accuracy | 0.994 |
| IoU | 0.577 |
| Dice | 0.621 |
| Precision | 0.643 |
| Recall | 0.676 |

Accuracy is not used as the main metric due to strong class imbalance.

Validation accuracy is slightly higher than training accuracy, likely due to:
- differences in tile distribution after spatial split
- class imbalance (dominance of background pixels)

## Example predictions

![Prediction examples](results/prediction_examples.png)

## Notes on data and environment

- The full project was developed and trained on a remote HPC server (SSH environment at NTNU) due to the large size of geospatial data (~TB scale) and computational requirements.
- This repository contains a lightweight version of the project:
  - No raw orthophotos or full datasets
  - Only code, configuration, and example outputs
- The pipeline remains fully reproducible with appropriate data access.

## How to run

```bash
pip install -r requirements.txt
python src/train.py