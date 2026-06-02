"""Multi-year inference for final road segmentation models.

The script applies:
- the color model to recent color years;
- the black-and-white model to historical years.

Outputs are binary predicted masks, one folder per year.

Run from the repository root with:
    python -m road_segmentation.inference_all_years
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import tensorflow as tf
from PIL import Image


# =========================================================
# 1. CONFIGURATION
# =========================================================
OLD_YEARS = [1947, 1957, 1964, 1977, 1991]
RECENT_YEARS = [
    1997, 2006, 2009, 2010, 2011,
    2014, 2015, 2016, 2017, 2019,
    2020, 2021, 2022,
]

DATASET_NAME = "data_512_no_stride"
PREDICTION_NAME = "final_model_test1"

INPUT_SIZE = (512, 512)
BATCH_SIZE = 16
OVERWRITE = False
MAX_IMAGES_PER_YEAR = None  # set to 100 for a quick test

# Use test-time augmentation for more stable predictions.
USE_TTA = False

DEFAULT_THRESHOLDS = {
    "color": 0.48,
    "black_and_white": 0.50,
}

# Optional threshold overrides by year. Fill only after manual/visual validation.
YEAR_THRESHOLD_OVERRIDES: dict[int, float] = {
    # 2019: 0.54,
    # 2021: 0.46,
}


# =========================================================
# 2. PATHS
# =========================================================
def find_project_dir() -> Path:
    env_project_dir = os.environ.get("MASTER_THESIS_PROJECT_DIR")
    if env_project_dir:
        return Path(env_project_dir).expanduser().resolve()

    file_path = Path(__file__).resolve()
    if file_path.parent.name == "road_segmentation" and file_path.parent.parent.name == "src":
        return file_path.parents[2]
    return Path.cwd().resolve()


PROJECT_DIR = find_project_dir()
DATASET_DIR = PROJECT_DIR / "data" / "03.pre_ML" / DATASET_NAME
FINAL_MODEL_ROOT_DIR = (
    PROJECT_DIR
    / "data"
    / "04.training"
    / "trondheim_training"
    / "final_models_2023_2024_2025_test1"
)
PREDICTION_ROOT_DIR = PROJECT_DIR / "data" / "05.predictions" / PREDICTION_NAME
PREDICTION_ROOT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = {
    "color": {
        "years": RECENT_YEARS,
        "image_folder": "images",
        "model_dir": FINAL_MODEL_ROOT_DIR / "color",
        "default_threshold": DEFAULT_THRESHOLDS["color"],
    },
    "black_and_white": {
        "years": OLD_YEARS,
        # Historical images are often already black-and-white but stored in images/.
        "image_folder": "images",
        "model_dir": FINAL_MODEL_ROOT_DIR / "black_and_white",
        "default_threshold": DEFAULT_THRESHOLDS["black_and_white"],
    },
}


# =========================================================
# 3. HELPERS
# =========================================================
def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def load_channel_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path) as f:
        data = json.load(f)
    return np.array(data["mean"], dtype=np.float32), np.array(data["std"], dtype=np.float32)


def load_threshold(path: Path, default_threshold: float) -> float:
    if not path.exists():
        print(f"[WARNING] Threshold file not found: {path}. Using {default_threshold}.")
        return float(default_threshold)

    with open(path) as f:
        data = json.load(f)

    for key in ["final_threshold", "best_threshold_for_area"]:
        if key in data:
            return float(data[key])

    print(f"[WARNING] Unknown threshold format in {path}. Using {default_threshold}.")
    return float(default_threshold)


def list_image_files(image_dir: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    if not image_dir.exists():
        print(f"[WARNING] Image directory does not exist: {image_dir}")
        return []
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions)


def load_and_normalize_image(image_path: Path, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize(INPUT_SIZE, resample=Image.BILINEAR)
    image = np.asarray(image).astype(np.float32) / 255.0
    image = (image - mean) / (std + 1e-6)
    return image


def save_binary_mask(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(output_path)


def get_model_paths(model_dir: Path) -> dict[str, Path]:
    return {
        "model_path": model_dir / "models" / "roads_extraction_final_2023_2024_2025.keras",
        "channel_stats_path": model_dir / "channel_stats.json",
        "threshold_path": model_dir / "final_threshold.json",
    }


def predict_batch_with_tta(model: tf.keras.Model, batch: np.ndarray) -> np.ndarray:
    """Predict probabilities with simple flip-based test-time augmentation."""
    batch_tf = tf.convert_to_tensor(batch, dtype=tf.float32)

    pred_original = model(batch_tf, training=False)

    batch_lr = tf.image.flip_left_right(batch_tf)
    pred_lr = tf.image.flip_left_right(model(batch_lr, training=False))

    batch_ud = tf.image.flip_up_down(batch_tf)
    pred_ud = tf.image.flip_up_down(model(batch_ud, training=False))

    batch_both = tf.image.flip_up_down(tf.image.flip_left_right(batch_tf))
    pred_both = model(batch_both, training=False)
    pred_both = tf.image.flip_left_right(tf.image.flip_up_down(pred_both))

    pred_mean = (pred_original + pred_lr + pred_ud + pred_both) / 4.0
    return pred_mean.numpy().squeeze(axis=-1)


# =========================================================
# 4. INFERENCE
# =========================================================
def predict_year(
    model: tf.keras.Model,
    year: int,
    image_folder: str,
    model_name: str,
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float,
) -> dict:
    start = perf_counter()
    image_dir = DATASET_DIR / str(year) / image_folder
    output_dir = PREDICTION_ROOT_DIR / str(year) / "masks_pred"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = list_image_files(image_dir)
    if MAX_IMAGES_PER_YEAR is not None:
        image_files = image_files[:MAX_IMAGES_PER_YEAR]

    year_threshold = YEAR_THRESHOLD_OVERRIDES.get(year, threshold)

    print("\n" + "=" * 80)
    print(f"Running inference for year {year}")
    print("=" * 80)
    print(f"Model:        {model_name}")
    print(f"Image folder: {image_folder}")
    print(f"Image dir:    {image_dir}")
    print(f"Output dir:   {output_dir}")
    print(f"Images:       {len(image_files)}")
    print(f"Threshold:    {year_threshold}")

    batch_images = []
    batch_output_paths = []
    n_predicted = 0
    n_skipped_existing = 0
    total_road_pixels = 0

    for index, image_path in enumerate(image_files, start=1):
        output_path = output_dir / f"{image_path.stem}.png"
        if output_path.exists() and not OVERWRITE:
            n_skipped_existing += 1
            continue

        batch_images.append(load_and_normalize_image(image_path, mean, std))
        batch_output_paths.append(output_path)

        should_run_batch = len(batch_images) == BATCH_SIZE or index == len(image_files)
        if should_run_batch and batch_images:
            batch = np.stack(batch_images, axis=0)
            probabilities = predict_batch_with_tta(model, batch) if USE_TTA else model.predict(batch, verbose=0).squeeze(axis=-1)

            for probability_map, mask_output_path in zip(probabilities, batch_output_paths):
                binary_mask = probability_map > year_threshold
                total_road_pixels += int(binary_mask.sum())
                save_binary_mask(binary_mask, mask_output_path)
                n_predicted += 1

            batch_images = []
            batch_output_paths = []

        if index == 1 or index % 500 == 0 or index == len(image_files):
            elapsed = perf_counter() - start
            progress = index / max(len(image_files), 1)
            eta = elapsed * (1.0 - progress) / progress if progress > 0 else 0
            print(
                f"[{index}/{len(image_files)}] predicted={n_predicted} "
                f"skipped_existing={n_skipped_existing} | elapsed={format_duration(elapsed)} | ETA={format_duration(eta)}"
            )

    elapsed = perf_counter() - start
    return {
        "year": year,
        "model": model_name,
        "image_folder": image_folder,
        "n_images": len(image_files),
        "n_predicted": n_predicted,
        "n_skipped_existing": n_skipped_existing,
        "road_pixels": int(total_road_pixels),
        "threshold": float(year_threshold),
        "output_dir": str(output_dir),
        "time_seconds": float(elapsed),
    }


def run_inference_for_model(model_name: str, config: dict) -> list[dict]:
    paths = get_model_paths(config["model_dir"])

    if not paths["model_path"].exists():
        raise FileNotFoundError(f"Model not found: {paths['model_path']}")
    if not paths["channel_stats_path"].exists():
        raise FileNotFoundError(f"Channel stats not found: {paths['channel_stats_path']}")

    print("\n" + "#" * 80)
    print(f"LOADING MODEL: {model_name}")
    print("#" * 80)
    print(f"Model path:         {paths['model_path']}")
    print(f"Channel stats path: {paths['channel_stats_path']}")
    print(f"Threshold path:     {paths['threshold_path']}")

    model = tf.keras.models.load_model(paths["model_path"], compile=False)
    mean, std = load_channel_stats(paths["channel_stats_path"])
    threshold = load_threshold(paths["threshold_path"], config["default_threshold"])

    results = []
    for year in config["years"]:
        results.append(
            predict_year(
                model=model,
                year=year,
                image_folder=config["image_folder"],
                model_name=model_name,
                mean=mean,
                std=std,
                threshold=threshold,
            )
        )

    tf.keras.backend.clear_session()
    return results


def main() -> None:
    total_start = perf_counter()

    print("\n" + "=" * 80)
    print("INFERENCE ON RECENT + OLD YEARS")
    print("=" * 80)
    print(f"Project dir:      {PROJECT_DIR}")
    print(f"Dataset dir:      {DATASET_DIR}")
    print(f"Prediction root:  {PREDICTION_ROOT_DIR}")
    print(f"Use TTA:          {USE_TTA}")
    print(f"Overwrite:        {OVERWRITE}")

    all_results = []
    for model_name, config in MODEL_CONFIGS.items():
        all_results.extend(run_inference_for_model(model_name, config))

    summary_path = PREDICTION_ROOT_DIR / "all_years_inference_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("ALL INFERENCE DONE")
    print("=" * 80)
    print(f"Summary saved to: {summary_path}")
    print(f"Total time:       {format_duration(perf_counter() - total_start)}")

    print("\nSummary by year:")
    for result in sorted(all_results, key=lambda x: x["year"]):
        print(
            f"{result['year']} | model={result['model']:<15s} | "
            f"images={result['n_images']:6d} | predicted={result['n_predicted']:6d} | "
            f"road_pixels={result['road_pixels']:12d} | threshold={result['threshold']:.3f}"
        )


if __name__ == "__main__":
    main()
