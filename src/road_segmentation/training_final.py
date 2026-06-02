"""Final training for road segmentation models.

This script trains the final color and black-and-white U-Net models on all
available labelled years, typically 2023, 2024 and 2025.

Unlike the experimental training script, this final training step uses no
validation set. Hyperparameters and thresholds should be selected beforehand
from experiments with validation/test splits.

Run from the repository root with:
    python -m road_segmentation.training_final

or directly:
    python src/road_segmentation/training_final.py
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

try:
    from road_segmentation.unet import unet
except ImportError:  # allows running this file directly from src/road_segmentation
    from unet import unet


# =========================================================
# 1. PARAMETERS
# =========================================================
INPUT_SIZE = (512, 512)
INPUT_SHAPE = (512, 512, 3)

FINAL_TRAIN_YEARS = [2023, 2024, 2025]
FINAL_EPOCHS = 14

# Thresholds selected from previous validation/test experiments.
FINAL_THRESHOLDS = {
    "color": 0.48,
    "black_and_white": 0.50,
}

EMPTY_KEEP_RATIO = 0.25
BATCH_SIZE = 16
SEED = 40
LEARNING_RATE = 5e-5
NORMALIZATION_SAMPLE_SIZE = 500

IMAGE_EXT = ".jpg"
MASK_EXT = ".png"
DATASET_NAME = "data_512_no_stride"

DEFAULT_THRESHOLD = 0.5

EXCLUDED_TILES = [
    "33-2-461-214-23",
    "33-2-461-214-32",
    "33-2-464-214-00",
]

EXPERIMENTS = {
    "color": {"image_key": "color_image_path"},
    "black_and_white": {"image_key": "bw_image_path"},
}

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =========================================================
# 2. PATHS
# =========================================================
def find_project_dir() -> Path:
    """Return the repository/project root directory."""
    env_project_dir = os.environ.get("MASTER_THESIS_PROJECT_DIR")
    if env_project_dir:
        return Path(env_project_dir).expanduser().resolve()

    file_path = Path(__file__).resolve()
    # src/road_segmentation/training_final.py -> repository root
    if file_path.parent.name == "road_segmentation" and file_path.parent.parent.name == "src":
        return file_path.parents[2]

    return Path.cwd().resolve()


PROJECT_DIR = find_project_dir()
DATASET_DIR = PROJECT_DIR / "data" / "03.pre_ML" / DATASET_NAME
OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "04.training"
    / "trondheim_training"
    / "final_models_2023_2024_2025_test1"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 3. GPU SETUP
# =========================================================
def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU is available" if gpus else "GPU is NOT available")


# =========================================================
# 4. FILE HELPERS
# =========================================================
def extract_tile_name_from_stem(stem: str) -> str:
    """Example: 33-2-461-214-22_0_2816 -> 33-2-461-214-22."""
    return "_".join(stem.split("_")[:-2])


def is_excluded_tile(stem: str, excluded_tiles: list[str]) -> bool:
    tile = extract_tile_name_from_stem(stem)
    return any(tile.startswith(excluded) for excluded in excluded_tiles)


def get_files_by_stem(folder: Path, extension: str) -> dict[str, Path]:
    if not folder.exists():
        print(f"[WARNING] Folder does not exist: {folder}")
        return {}

    return {path.stem: path for path in folder.glob(f"*{extension}") if path.is_file()}


# =========================================================
# 5. SAMPLE TABLE
# =========================================================
def build_master_samples(
    dataset_dir: Path,
    years: list[int],
    image_ext: str = ".jpg",
    mask_ext: str = ".png",
    excluded_tiles: Optional[list[str]] = None,
) -> list[dict]:
    """Build a unified sample table for all labelled years.

    A sample is included only when the color image, black-and-white image, and
    mask all exist. This keeps the color and black-and-white experiments fully
    comparable.
    """
    if excluded_tiles is None:
        excluded_tiles = []

    samples: list[dict] = []

    for year in years:
        year_dir = dataset_dir / str(year)

        color_map = get_files_by_stem(year_dir / "images", image_ext)
        bw_map = get_files_by_stem(year_dir / "images_b_and_w", image_ext)
        mask_map = get_files_by_stem(year_dir / "masks", mask_ext)
        common_stems = sorted(set(color_map) & set(bw_map) & set(mask_map))

        excluded_count = 0
        for stem in common_stems:
            if is_excluded_tile(stem, excluded_tiles):
                excluded_count += 1
                continue

            samples.append(
                {
                    "year": year,
                    "stem": stem,
                    "tile_name": extract_tile_name_from_stem(stem),
                    "color_image_path": color_map[stem],
                    "bw_image_path": bw_map[stem],
                    "mask_path": mask_map[stem],
                }
            )

        print("\n" + "=" * 80)
        print(f"Year {year}")
        print(f"Color images:        {len(color_map)}")
        print(f"Black/white images:  {len(bw_map)}")
        print(f"Masks:               {len(mask_map)}")
        print(f"Matched triplets:    {len(common_stems)}")
        print(f"Excluded stems:      {excluded_count}")

    print("\n" + "=" * 80)
    print(f"Total samples after tile exclusion: {len(samples)}")
    return samples


def filter_samples_by_mask_content(
    samples: list[dict],
    empty_keep_ratio: float = 0.25,
    seed: int = 42,
) -> list[dict]:
    """Keep all positive masks and a random fraction of empty masks."""
    rng = np.random.default_rng(seed)

    kept: list[dict] = []
    positive_count = 0
    empty_count = 0
    kept_empty_count = 0
    unreadable_count = 0

    for sample in samples:
        mask = cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            unreadable_count += 1
            continue

        if np.any(mask > 0):
            kept.append(sample)
            positive_count += 1
        else:
            empty_count += 1
            if rng.random() < empty_keep_ratio:
                kept.append(sample)
                kept_empty_count += 1

    print("\n=== Filtering summary ===")
    print(f"Positive patches kept: {positive_count}")
    print(f"Empty patches seen:    {empty_count}")
    print(f"Empty patches kept:    {kept_empty_count}")
    print(f"Unreadable masks:      {unreadable_count}")
    print(f"Final sample count:    {len(kept)}")
    return kept


# =========================================================
# 6. NORMALIZATION AND DATA PIPELINE
# =========================================================
def compute_channel_stats(
    samples: list[dict],
    image_key: str,
    n: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean/std on a random subset of training images."""
    if not samples:
        raise ValueError("Cannot compute channel stats from an empty sample list.")

    rng = np.random.default_rng(seed)
    subset_indices = rng.choice(len(samples), size=min(n, len(samples)), replace=False)

    means = []
    stds = []
    for idx in subset_indices:
        img = cv2.imread(str(samples[idx][image_key]))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        means.append(img.mean(axis=(0, 1)))
        stds.append(img.std(axis=(0, 1)))

    if not means:
        raise RuntimeError("Could not compute channel stats: no readable images.")

    mean = np.mean(means, axis=0).astype(np.float32)
    std = np.mean(stds, axis=0).astype(np.float32)

    print(f"Channel stats computed on {len(means)} images")
    print(f"Mean: {np.round(mean, 4)}")
    print(f"Std:  {np.round(std, 4)}")
    return mean, std


def save_channel_stats(mean: np.ndarray, std: np.ndarray, output_path: Path) -> None:
    with open(output_path, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)
    print(f"Channel stats saved -> {output_path}")


AUTOTUNE = tf.data.AUTOTUNE


def load_image_mask(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, INPUT_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, INPUT_SIZE, method="nearest")
    mask = tf.cast(mask > 0, tf.float32)

    return image, mask


@tf.function
def augment_before_normalization(image, mask):
    """Apply augmentation while images are still in [0, 1]."""
    seed_lr = tf.random.uniform(shape=(2,), maxval=2**31 - 1, dtype=tf.int32)
    seed_ud = tf.random.uniform(shape=(2,), maxval=2**31 - 1, dtype=tf.int32)

    image = tf.image.stateless_random_flip_left_right(image, seed=seed_lr)
    mask = tf.image.stateless_random_flip_left_right(mask, seed=seed_lr)

    image = tf.image.stateless_random_flip_up_down(image, seed=seed_ud)
    mask = tf.image.stateless_random_flip_up_down(mask, seed=seed_ud)

    k = tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)
    mask = tf.image.rot90(mask, k=k)

    # Light photometric augmentation for final training.
    image = tf.image.random_brightness(image, max_delta=0.12)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, mask


def make_normalize_fn(mean: np.ndarray, std: np.ndarray):
    mean_t = tf.constant(mean, dtype=tf.float32)
    std_t = tf.constant(std + 1e-6, dtype=tf.float32)

    def normalize(image, mask):
        return (image - mean_t) / std_t, mask

    return normalize


def make_dataset(
    samples: list[dict],
    image_key: str,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    image_paths = [str(s[image_key]) for s in samples]
    mask_paths = [str(s["mask_path"]) for s in samples]

    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    if shuffle:
        ds = ds.shuffle(min(len(image_paths), 10_000), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(load_image_mask, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.map(augment_before_normalization, num_parallel_calls=AUTOTUNE)
    ds = ds.map(make_normalize_fn(mean, std), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


# =========================================================
# 7. LOSS AND METRICS
# =========================================================
def dice_loss(y_true, y_pred, smooth: float = 1e-6):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(tf.cast(y_pred, tf.float32))
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1.0 - dice


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)


class DiceCoef(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, smooth=1e-6, name="dice_coef", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.smooth = smooth
        self.intersection = self.add_weight(name="intersection", initializer="zeros")
        self.sum_true = self.add_weight(name="sum_true", initializer="zeros")
        self.sum_pred = self.add_weight(name="sum_pred", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred > self.threshold, [-1]), tf.float32)
        self.intersection.assign_add(tf.reduce_sum(y_true * y_pred))
        self.sum_true.assign_add(tf.reduce_sum(y_true))
        self.sum_pred.assign_add(tf.reduce_sum(y_pred))

    def result(self):
        return (2.0 * self.intersection + self.smooth) / (self.sum_true + self.sum_pred + self.smooth)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.sum_true.assign(0.0)
        self.sum_pred.assign(0.0)


class IoUScore(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, smooth=1e-6, name="iou_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.smooth = smooth
        self.intersection = self.add_weight(name="intersection", initializer="zeros")
        self.union = self.add_weight(name="union", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred > self.threshold, [-1]), tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        self.intersection.assign_add(intersection)
        self.union.assign_add(union)

    def result(self):
        return (self.intersection + self.smooth) / (self.union + self.smooth)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)


class AreaBias(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name="area_bias", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.sum_pred = self.add_weight(name="sum_pred", initializer="zeros")
        self.sum_true = self.add_weight(name="sum_true", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred > self.threshold, [-1]), tf.float32)
        self.sum_pred.assign_add(tf.reduce_sum(y_pred))
        self.sum_true.assign_add(tf.reduce_sum(y_true))

    def result(self):
        return (self.sum_pred - self.sum_true) / (self.sum_true + 1e-6)

    def reset_state(self):
        self.sum_pred.assign(0.0)
        self.sum_true.assign(0.0)


class RelativeAreaError(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name="rae", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.sum_pred = self.add_weight(name="sum_pred", initializer="zeros")
        self.sum_true = self.add_weight(name="sum_true", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred > self.threshold, [-1]), tf.float32)
        self.sum_pred.assign_add(tf.reduce_sum(y_pred))
        self.sum_true.assign_add(tf.reduce_sum(y_true))

    def result(self):
        return tf.abs(self.sum_pred - self.sum_true) / (self.sum_true + 1e-6)

    def reset_state(self):
        self.sum_pred.assign(0.0)
        self.sum_true.assign(0.0)


def build_model() -> tf.keras.Model:
    model = unet(INPUT_SHAPE, output_layer=1)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=bce_dice_loss,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            DiceCoef(threshold=DEFAULT_THRESHOLD),
            IoUScore(threshold=DEFAULT_THRESHOLD),
            tf.keras.metrics.Precision(name="precision", thresholds=DEFAULT_THRESHOLD),
            tf.keras.metrics.Recall(name="recall", thresholds=DEFAULT_THRESHOLD),
            AreaBias(threshold=DEFAULT_THRESHOLD),
            RelativeAreaError(threshold=DEFAULT_THRESHOLD),
        ],
    )
    return model


# =========================================================
# 8. FINAL TRAINING
# =========================================================
def save_history_plots(history, plot_dir: Path, experiment_name: str) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    epochs_ran = range(1, len(history.history["loss"]) + 1)

    for metric_name in ["loss", "dice_coef", "iou_score", "rae", "area_bias", "precision", "recall"]:
        if metric_name not in history.history:
            continue
        plt.figure(figsize=(8, 5))
        plt.plot(epochs_ran, history.history[metric_name], label=f"Train {metric_name}")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.title(f"{experiment_name} final training - {metric_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{experiment_name}_final_train_{metric_name}.png", bbox_inches="tight", dpi=150)
        plt.close()


def train_final_experiment(
    experiment_name: str,
    image_key: str,
    train_samples: list[dict],
    experiment_dir: Path,
    final_epochs: int,
    final_threshold: float,
) -> dict:
    print("\n" + "=" * 80)
    print(f"FINAL TRAINING EXPERIMENT: {experiment_name}")
    print("=" * 80)

    model_dir = experiment_dir / "models"
    plot_dir = experiment_dir / "plots"
    model_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    counts = Counter(s["year"] for s in train_samples)
    print(f"Train samples: {len(train_samples)}")
    for year in sorted(counts):
        print(f"  {year}: {counts[year]}")

    mean, std = compute_channel_stats(train_samples, image_key, n=NORMALIZATION_SAMPLE_SIZE, seed=SEED)
    channel_stats_path = experiment_dir / "channel_stats.json"
    save_channel_stats(mean, std, channel_stats_path)

    train_ds = make_dataset(train_samples, image_key, mean, std, BATCH_SIZE, shuffle=True)

    model = build_model()
    model.summary()

    history = model.fit(train_ds, epochs=final_epochs)

    final_model_path = model_dir / "roads_extraction_final_2023_2024_2025.keras"
    model.save(str(final_model_path))
    print(f"Model saved -> {final_model_path}")

    threshold_path = experiment_dir / "final_threshold.json"
    with open(threshold_path, "w") as f:
        json.dump(
            {
                "final_threshold": final_threshold,
                "source": "Previous area-calibrated validation/test experiment",
                "note": "No validation set is used during final training.",
            },
            f,
            indent=2,
        )

    history_path = experiment_dir / "final_training_history.json"
    with open(history_path, "w") as f:
        json.dump({key: [float(v) for v in values] for key, values in history.history.items()}, f, indent=2)

    save_history_plots(history, plot_dir, experiment_name)
    tf.keras.backend.clear_session()

    return {
        "experiment": experiment_name,
        "n_train_samples": len(train_samples),
        "samples_per_year": {str(year): int(counts[year]) for year in sorted(counts)},
        "final_epochs": final_epochs,
        "final_threshold": final_threshold,
        "model_path": str(final_model_path),
        "channel_stats_path": str(channel_stats_path),
        "threshold_path": str(threshold_path),
        "history_path": str(history_path),
    }


def main() -> None:
    configure_gpu()

    print("\n" + "=" * 80)
    print("FINAL TRAINING ON 2023 + 2024 + 2025")
    print("=" * 80)
    print(f"Project dir:      {PROJECT_DIR}")
    print(f"Dataset dir:      {DATASET_DIR}")
    print(f"Output dir:       {OUTPUT_DIR}")
    print(f"Training years:   {FINAL_TRAIN_YEARS}")
    print(f"Final epochs:     {FINAL_EPOCHS}")

    all_samples = build_master_samples(DATASET_DIR, FINAL_TRAIN_YEARS, IMAGE_EXT, MASK_EXT, EXCLUDED_TILES)
    final_train_samples = filter_samples_by_mask_content(all_samples, EMPTY_KEEP_RATIO, SEED)

    final_samples_path = OUTPUT_DIR / "final_train_samples.json"
    with open(final_samples_path, "w") as f:
        json.dump(
            [
                {key: str(value) if isinstance(value, Path) else value for key, value in sample.items()}
                for sample in final_train_samples
            ],
            f,
            indent=2,
        )
    print(f"Final sample list saved -> {final_samples_path}")

    all_results = {}
    for experiment_name, experiment_config in EXPERIMENTS.items():
        if experiment_name not in FINAL_THRESHOLDS:
            raise ValueError(f"No final threshold defined for experiment: {experiment_name}")
        all_results[experiment_name] = train_final_experiment(
            experiment_name=experiment_name,
            image_key=experiment_config["image_key"],
            train_samples=final_train_samples,
            experiment_dir=OUTPUT_DIR / experiment_name,
            final_epochs=FINAL_EPOCHS,
            final_threshold=FINAL_THRESHOLDS[experiment_name],
        )

    final_results_path = OUTPUT_DIR / "final_training_results.json"
    with open(final_results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Final results saved -> {final_results_path}")


if __name__ == "__main__":
    main()
