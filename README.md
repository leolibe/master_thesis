# Master Thesis – Leo Liberkowski  
### Road Segmentation from Norwegian Orthophotos

This project implements a deep learning pipeline for **road segmentation** using aerial orthophotos and vector road data.

---

## Objective

The goal is to generate **binary road masks** from geospatial data and train a **U-Net model** to automatically segment roads from high-resolution orthophotos.

---

## Project Structure


master_thesis/
├── configs/train_config.yaml # Training hyperparameters
├── notebooks/demo_predictions.ipynb
├── src/road_segmentation/
│ ├── mask_generation.py # GML → PNG masks
│ ├── patch_generation.py # Tiles → 512×512 patches
│ ├── preprocessing.py # Filtering + spatial split
│ ├── dataset.py # tf.data pipeline
│ ├── train.py # U-Net training
│ ├── evaluate.py # Test metrics
│ └── visualization.py # Prediction plots
└── results/


---

## Data

- **Orthophotos**: GeoTIFF (30 cm/pixel resolution)  
- **Coordinate system**: EPSG:25833  
- **Road data**: GML (vector format)  
- **Generated masks**: Binary PNG images  

Raw data is **not included** due to size and licensing constraints (NTNU datasets).

---

## Pipeline

1. Load orthophotos and GML road geometries  
2. Reproject geometries to EPSG:25833  
3. Rasterize roads into binary masks  
4. Generate aligned image/mask patches (512×512)  
5. Filter empty patches  
6. Split dataset by tile (to avoid spatial leakage)  
7. Train a U-Net segmentation model  
8. Evaluate using Dice, IoU, Precision, and Recall

## Pipeline overview
![Pipeline overview](docs/pipeline_overview.JPG)

---

## Key Technical Choices

- **Tile-based split**  
  Prevents spatial leakage: nearby patches must not appear in both training and validation sets, as this would artificially inflate performance.

- **BCE + Dice Loss**  
  Roads represent only ~3% of pixels.  
  - Binary Cross-Entropy alone struggles with class imbalance  
  - Dice loss directly penalizes false negatives  

- **Empty patch filtering (`empty_keep_ratio=0.25`)**  
  Keeps 25% of patches without roads so the model learns to correctly identify background.

---

## Results

Example metrics on the test set:

| Metric     | Value |
|------------|------|
| Accuracy   | 0.994 |
| IoU        | 0.577 |
| Dice       | 0.621 |
| Precision  | 0.643 |
| Recall     | 0.676 |

Accuracy is not a reliable metric here due to **strong class imbalance**.

Validation accuracy may exceed training accuracy due to:
- spatial distribution differences after tile-based split  
- dominance of background pixels  

---

## Example Predictions

![Prediction examples](results/prediction_examples.png)

---

## Requirements

- Python 3.10+  
- Conda (recommended for geospatial dependencies)  
- Access to orthophotos from GeoNorge (not provided)  

---

## Installation & Usage

```bash
# Create environment
conda create -n roads python=3.10
conda activate roads

# Install dependencies
pip install -r requirements.txt

# Data preparation
python -m src.road_segmentation.mask_generation --tif-dir data/tifs --gml data/roads.gml
python -m src.road_segmentation.patch_generation --image-dir data/tifs --mask-dir data/masks

# Train model
python -m src.road_segmentation.train
