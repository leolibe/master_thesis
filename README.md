# Master Thesis - Road Segmentation from Norwegian Orthophotos
Code used for the Master thesis of Leo Liberkowski (student at NTNU): "Machine Learning–Based Mapping of Norway’s Road Network from Orthophotos for Quantification of the Stock of the Frost Protection Layer"

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

## Example predictions

![Prediction examples](results/prediction_examples.png)

## How to run

```bash
pip install -r requirements.txt
python src/train.py

---

## `.gitignore` important

Crée un fichier `.gitignore` :

```gitignore
# Python
__pycache__/
*.pyc
.ipynb_checkpoints/

# Environments
.env
.venv/
venv/
*.conda

# Data
data/
*.tif
*.tiff
*.gml
*.zip
*.npz
*.npy

# Models
*.h5
*.keras
*.pth
*.pt

# Large outputs
outputs/
checkpoints/
logs/

# OS
.DS_Store