"""Post-processing and road-width area classification.

Input:
    data/05.predictions/<PREDICTION_NAME>/<YEAR>/masks_pred/*.png

Outputs:
    - cleaned masks per year
    - width maps per patch (.npy)
    - class masks per patch (.png)
    - patch/component/year summary CSV files
    - area_by_year.json
    - area evolution plots

The width classification uses a distance transform and skeleton-based width
estimation per connected road component.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

try:
    from skimage.morphology import skeletonize as skimage_skeletonize
except Exception:  # pragma: no cover
    skimage_skeletonize = None


# =========================================================
# 1. CONFIGURATION
# =========================================================
YEARS = [
    1947, 1957, 1964, 1977, 1991,
    1997, 2006, 2009, 2010, 2011,
    2014, 2015, 2016, 2017, 2019,
    2020, 2021, 2022,
]

PREDICTION_NAME = "final_model_test1"
INPUT_MASK_FOLDER = "masks_pred"
OUTPUT_CLEAN_FOLDER = "masks_postprocessed"
OUTPUT_WIDTH_FOLDER = "width_maps"
OUTPUT_CLASS_FOLDER = "class_masks"

# IMPORTANT: set this to the true spatial resolution of the patch images.
# Example: 0.25 means one pixel represents 0.25 m x 0.25 m.
PIXEL_SIZE_M = 0.25

WIDTH_CLASS_NAMES = {
    1: "0_4m",
    2: "4_6m",
    3: "6_8m",
    4: "8m_plus",
}

OVERWRITE = False
MAX_MASKS_PER_YEAR = None  # set to 100 for quick tests

# Default post-processing. Keep conservative values because this pipeline is
# used for road area estimation.
DEFAULT_POSTPROCESSING_CONFIG = {
    "closing_kernel_size": 5,
    "closing_iterations": 1,
    "opening_kernel_size": 3,
    "opening_iterations": 1,
    "min_component_area_pixels": 50,
    "max_hole_area_pixels": 150,
}

# More conservative settings for years with snow/noise/unstable predictions.
YEAR_POSTPROCESSING_CONFIG = {
    1977: {
        "closing_kernel_size": 3,
        "closing_iterations": 1,
        "opening_kernel_size": 3,
        "opening_iterations": 1,
        "min_component_area_pixels": 120,
        "max_hole_area_pixels": 80,
    },
    2011: {
        "closing_kernel_size": 3,
        "closing_iterations": 1,
        "opening_kernel_size": 3,
        "opening_iterations": 1,
        "min_component_area_pixels": 120,
        "max_hole_area_pixels": 80,
    },
    2016: {
        "closing_kernel_size": 3,
        "closing_iterations": 1,
        "opening_kernel_size": 3,
        "opening_iterations": 1,
        "min_component_area_pixels": 120,
        "max_hole_area_pixels": 80,
    },
    2019: {
        "closing_kernel_size": 3,
        "closing_iterations": 1,
        "opening_kernel_size": 3,
        "opening_iterations": 1,
        "min_component_area_pixels": 120,
        "max_hole_area_pixels": 80,
    },
}

MIN_SKELETON_PIXELS = 5


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
PREDICTION_ROOT_DIR = PROJECT_DIR / "data" / "05.predictions" / PREDICTION_NAME


# =========================================================
# 3. HELPERS
# =========================================================
def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_postprocessing_config(year: int) -> dict:
    config = DEFAULT_POSTPROCESSING_CONFIG.copy()
    config.update(YEAR_POSTPROCESSING_CONFIG.get(year, {}))
    return config


def load_binary_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")
    return (mask > 0).astype(np.uint8)


def save_binary_mask(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(output_path)


def save_class_mask(class_mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(class_mask.astype(np.uint8)).save(output_path)


def remove_small_components(mask: np.ndarray, min_area_pixels: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area_pixels:
            cleaned[labels == label] = 1
    return cleaned


def fill_small_holes(mask: np.ndarray, max_hole_area_pixels: int) -> np.ndarray:
    mask = mask.astype(np.uint8)
    inverted = 1 - mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    filled = mask.copy()
    height, width = mask.shape

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > max_hole_area_pixels:
            continue

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        touches_border = x == 0 or y == 0 or x + w >= width or y + h >= height
        if not touches_border:
            filled[labels == label] = 1

    return filled


def postprocess_mask(mask: np.ndarray, config: dict) -> np.ndarray:
    """Close small gaps, fill small holes, and remove isolated noise."""
    mask = mask.astype(np.uint8)

    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config["closing_kernel_size"], config["closing_kernel_size"]),
    )
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel, iterations=config["closing_iterations"])

    filled = fill_small_holes(closed, max_hole_area_pixels=config["max_hole_area_pixels"])

    opening_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config["opening_kernel_size"], config["opening_kernel_size"]),
    )
    opened = cv2.morphologyEx(filled, cv2.MORPH_OPEN, opening_kernel, iterations=config["opening_iterations"])

    cleaned = remove_small_components(opened, min_area_pixels=config["min_component_area_pixels"])
    return cleaned.astype(np.uint8)


def skeletonize_binary(mask: np.ndarray) -> np.ndarray:
    """Skeletonize a binary mask. Uses scikit-image when available."""
    if skimage_skeletonize is not None:
        return skimage_skeletonize(mask > 0).astype(np.uint8)

    # OpenCV fallback.
    work = (mask > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(work, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(work, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(work, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        work = eroded.copy()
        if cv2.countNonZero(work) == 0:
            break
    return (skeleton > 0).astype(np.uint8)


def compute_width_map(mask: np.ndarray, pixel_size_m: float) -> np.ndarray:
    distance_pixels = cv2.distanceTransform(mask.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5)
    width_m = 2.0 * distance_pixels * pixel_size_m
    width_m[mask == 0] = 0.0
    return width_m.astype(np.float32)


def width_to_class(width_m: float) -> int:
    if width_m < 4.0:
        return 1
    if width_m < 6.0:
        return 2
    if width_m < 8.0:
        return 3
    return 4


def classify_components_by_width(mask: np.ndarray, width_map_m: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Classify each connected road component by median skeleton width."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    skeleton = skeletonize_binary(mask)

    class_mask = np.zeros_like(mask, dtype=np.uint8)
    component_rows: list[dict] = []

    for label in range(1, num_labels):
        component_mask = labels == label
        area_pixels = int(stats[label, cv2.CC_STAT_AREA])

        component_skeleton = skeleton.astype(bool) & component_mask
        skeleton_widths = width_map_m[component_skeleton]

        if len(skeleton_widths) >= MIN_SKELETON_PIXELS:
            estimated_width_m = float(np.median(skeleton_widths))
            mean_width_m = float(np.mean(skeleton_widths))
            p90_width_m = float(np.percentile(skeleton_widths, 90))
            n_skeleton_pixels = int(len(skeleton_widths))
        else:
            component_widths = width_map_m[component_mask]
            estimated_width_m = float(np.percentile(component_widths, 90)) if len(component_widths) else 0.0
            mean_width_m = float(np.mean(component_widths)) if len(component_widths) else 0.0
            p90_width_m = float(np.percentile(component_widths, 90)) if len(component_widths) else 0.0
            n_skeleton_pixels = int(len(skeleton_widths))

        class_id = width_to_class(estimated_width_m)
        class_mask[component_mask] = class_id

        component_rows.append(
            {
                "component_id": int(label),
                "area_pixels": area_pixels,
                "estimated_width_m": estimated_width_m,
                "mean_skeleton_width_m": mean_width_m,
                "p90_skeleton_width_m": p90_width_m,
                "n_skeleton_pixels": n_skeleton_pixels,
                "width_class_id": int(class_id),
                "width_class": WIDTH_CLASS_NAMES[class_id],
            }
        )

    return class_mask, component_rows


def compute_patch_area_stats(original_mask: np.ndarray, clean_mask: np.ndarray, class_mask: np.ndarray) -> dict:
    pixel_area_m2 = PIXEL_SIZE_M * PIXEL_SIZE_M
    original_pixels = int(original_mask.sum())
    clean_pixels = int(clean_mask.sum())

    stats = {
        "original_road_pixels": original_pixels,
        "postprocessed_road_pixels": clean_pixels,
        "difference_pixels": clean_pixels - original_pixels,
        "relative_difference": (clean_pixels - original_pixels) / (original_pixels + 1e-6),
        "total_area_m2": clean_pixels * pixel_area_m2,
        "total_area_ha": clean_pixels * pixel_area_m2 / 10_000.0,
    }

    for class_id, class_name in WIDTH_CLASS_NAMES.items():
        pixels = int(np.sum(class_mask == class_id))
        stats[f"pixels_{class_name}"] = pixels
        stats[f"area_m2_{class_name}"] = pixels * pixel_area_m2
        stats[f"area_ha_{class_name}"] = pixels * pixel_area_m2 / 10_000.0

    return stats


# =========================================================
# 4. PROCESSING
# =========================================================
def process_year(year: int) -> tuple[list[dict], list[dict]]:
    start = perf_counter()
    input_dir = PREDICTION_ROOT_DIR / str(year) / INPUT_MASK_FOLDER
    clean_dir = PREDICTION_ROOT_DIR / str(year) / OUTPUT_CLEAN_FOLDER
    width_dir = PREDICTION_ROOT_DIR / str(year) / OUTPUT_WIDTH_FOLDER
    class_dir = PREDICTION_ROOT_DIR / str(year) / OUTPUT_CLASS_FOLDER

    clean_dir.mkdir(parents=True, exist_ok=True)
    width_dir.mkdir(parents=True, exist_ok=True)
    class_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"[WARNING] No prediction folder for year {year}: {input_dir}")
        return [], []

    mask_files = sorted(input_dir.glob("*.png"))
    if MAX_MASKS_PER_YEAR is not None:
        mask_files = mask_files[:MAX_MASKS_PER_YEAR]

    config = get_postprocessing_config(year)
    print("\n" + "=" * 80)
    print(f"Post-processing year {year}")
    print("=" * 80)
    print(f"Input dir:  {input_dir}")
    print(f"Masks:      {len(mask_files)}")
    print(f"Config:     {config}")

    patch_rows: list[dict] = []
    component_rows_all: list[dict] = []

    for index, mask_path in enumerate(mask_files, start=1):
        clean_path = clean_dir / mask_path.name
        width_path = width_dir / f"{mask_path.stem}.npy"
        class_path = class_dir / mask_path.name

        if clean_path.exists() and width_path.exists() and class_path.exists() and not OVERWRITE:
            continue

        original_mask = load_binary_mask(mask_path)
        clean_mask = postprocess_mask(original_mask, config=config)
        width_map_m = compute_width_map(clean_mask, PIXEL_SIZE_M)
        class_mask, component_rows = classify_components_by_width(clean_mask, width_map_m)

        save_binary_mask(clean_mask, clean_path)
        np.save(width_path, width_map_m)
        save_class_mask(class_mask, class_path)

        patch_stats = compute_patch_area_stats(original_mask, clean_mask, class_mask)
        patch_stats.update(
            {
                "year": year,
                "patch_name": mask_path.stem,
                "mask_path": str(mask_path),
                "clean_mask_path": str(clean_path),
                "width_map_path": str(width_path),
                "class_mask_path": str(class_path),
            }
        )
        patch_rows.append(patch_stats)

        for row in component_rows:
            row.update({"year": year, "patch_name": mask_path.stem})
            component_rows_all.append(row)

        if index == 1 or index % 500 == 0 or index == len(mask_files):
            elapsed = perf_counter() - start
            progress = index / max(len(mask_files), 1)
            eta = elapsed * (1.0 - progress) / progress if progress else 0
            print(f"[{index}/{len(mask_files)}] {mask_path.name} | elapsed={format_duration(elapsed)} | ETA={format_duration(eta)}")

    print(f"Done year {year} in {format_duration(perf_counter() - start)}")
    return patch_rows, component_rows_all


def build_year_summary(patch_df: pd.DataFrame) -> pd.DataFrame:
    agg_dict = {
        "patch_name": "count",
        "original_road_pixels": "sum",
        "postprocessed_road_pixels": "sum",
        "difference_pixels": "sum",
        "total_area_m2": "sum",
        "total_area_ha": "sum",
    }
    for class_name in WIDTH_CLASS_NAMES.values():
        agg_dict[f"pixels_{class_name}"] = "sum"
        agg_dict[f"area_m2_{class_name}"] = "sum"
        agg_dict[f"area_ha_{class_name}"] = "sum"

    year_df = patch_df.groupby("year").agg(agg_dict).rename(columns={"patch_name": "n_patches"}).reset_index()
    year_df["relative_difference_after_postprocessing"] = (
        year_df["postprocessed_road_pixels"] - year_df["original_road_pixels"]
    ) / (year_df["original_road_pixels"] + 1e-6)
    return year_df


def save_area_by_year_json(year_df: pd.DataFrame, output_path: Path) -> None:
    data = {}
    for _, row in year_df.iterrows():
        year = int(row["year"])
        data[str(year)] = {
            "total_area_m2": float(row["total_area_m2"]),
            "total_area_ha": float(row["total_area_ha"]),
            "area_by_width_m2": {
                class_name: float(row[f"area_m2_{class_name}"])
                for class_name in WIDTH_CLASS_NAMES.values()
            },
            "area_by_width_ha": {
                class_name: float(row[f"area_ha_{class_name}"])
                for class_name in WIDTH_CLASS_NAMES.values()
            },
            "postprocessing": {
                "original_road_pixels": int(row["original_road_pixels"]),
                "postprocessed_road_pixels": int(row["postprocessed_road_pixels"]),
                "relative_difference_after_postprocessing": float(row["relative_difference_after_postprocessing"]),
            },
        }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON -> {output_path}")


def plot_area_evolution(year_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(11, 6))
    for class_name, label in [("0_4m", "0-4 m"), ("4_6m", "4-6 m"), ("6_8m", "6-8 m"), ("8m_plus", ">8 m")]:
        plt.plot(year_df["year"], year_df[f"area_ha_{class_name}"], marker="o", label=label)
    plt.xlabel("Year")
    plt.ylabel("Road area [ha]")
    plt.title("Evolution of road area by width category")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved plot -> {output_path}")


def plot_total_area_evolution(year_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(11, 6))
    plt.plot(year_df["year"], year_df["total_area_ha"], marker="o", label="Total road area")
    plt.xlabel("Year")
    plt.ylabel("Road area [ha]")
    plt.title("Evolution of total road area")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved plot -> {output_path}")


def main() -> None:
    total_start = perf_counter()
    print("\n" + "=" * 80)
    print("POST-PROCESSING: GAP FILLING + WIDTH CLASSIFICATION")
    print("=" * 80)
    print(f"Project dir:       {PROJECT_DIR}")
    print(f"Prediction root:   {PREDICTION_ROOT_DIR}")
    print(f"Pixel size [m]:    {PIXEL_SIZE_M}")

    all_patch_rows: list[dict] = []
    all_component_rows: list[dict] = []

    for year in YEARS:
        patch_rows, component_rows = process_year(year)
        all_patch_rows.extend(patch_rows)
        all_component_rows.extend(component_rows)

    if not all_patch_rows:
        print("No masks were processed.")
        return

    patch_df = pd.DataFrame(all_patch_rows)
    component_df = pd.DataFrame(all_component_rows)
    year_df = build_year_summary(patch_df)

    patch_stats_path = PREDICTION_ROOT_DIR / "postprocessing_patch_stats.csv"
    component_stats_path = PREDICTION_ROOT_DIR / "postprocessing_component_stats.csv"
    year_stats_path = PREDICTION_ROOT_DIR / "postprocessing_year_stats.csv"
    area_json_path = PREDICTION_ROOT_DIR / "area_by_year.json"
    area_plot_path = PREDICTION_ROOT_DIR / "area_evolution_by_width.png"
    total_area_plot_path = PREDICTION_ROOT_DIR / "area_evolution_total.png"

    patch_df.to_csv(patch_stats_path, index=False)
    component_df.to_csv(component_stats_path, index=False)
    year_df.to_csv(year_stats_path, index=False)
    save_area_by_year_json(year_df, area_json_path)
    plot_area_evolution(year_df, area_plot_path)
    plot_total_area_evolution(year_df, total_area_plot_path)

    print("\nSaved outputs:")
    print(f"Patch stats:      {patch_stats_path}")
    print(f"Component stats:  {component_stats_path}")
    print(f"Year stats:       {year_stats_path}")
    print(f"Area JSON:        {area_json_path}")
    print(f"Area plot:        {area_plot_path}")
    print(f"Total area plot:  {total_area_plot_path}")
    print(f"Total time:       {format_duration(perf_counter() - total_start)}")


if __name__ == "__main__":
    main()
