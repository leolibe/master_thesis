from pathlib import Path
import numpy as np
from collections import Counter


# =========================================================
# 1. BASIC TRANSFORMS
# =========================================================
def normalize_image(image):
    """
    Normalize image to [0, 1]
    """
    return image.astype(np.float32) / 255.0


def binarize_mask(mask):
    """
    Convert mask to binary (0 / 1)
    """
    return (mask > 0).astype(np.uint8)


# =========================================================
# 2. FILTER PATCHES (CLASS IMBALANCE)
# =========================================================
def filter_by_mask_content(
    images,
    masks,
    empty_keep_ratio=0.25,
    seed=42
):
    """
    Keep:
    - all patches containing at least one positive pixel
    - only a fraction of empty patches

    Returns filtered arrays and kept indices
    """

    if len(images) != len(masks):
        raise ValueError("images and masks must have same length")

    rng = np.random.default_rng(seed)

    # Flatten masks to detect positive pixels
    masks_flat = masks.reshape(len(masks), -1)

    positive_mask = np.any(masks_flat > 0, axis=1)
    empty_mask = ~positive_mask

    pos_idx = np.where(positive_mask)[0]
    empty_idx = np.where(empty_mask)[0]

    n_keep_empty = int(len(empty_idx) * empty_keep_ratio)

    if n_keep_empty > 0:
        kept_empty_idx = rng.choice(empty_idx, size=n_keep_empty, replace=False)
    else:
        kept_empty_idx = np.array([], dtype=int)

    kept_idx = np.concatenate([pos_idx, kept_empty_idx])
    kept_idx = np.sort(kept_idx)

    print("=== Filtering summary ===")
    print(f"Total: {len(images)}")
    print(f"Positive: {len(pos_idx)}")
    print(f"Empty: {len(empty_idx)}")
    print(f"Empty kept: {len(kept_empty_idx)}")
    print(f"Final: {len(kept_idx)}")

    return images[kept_idx], masks[kept_idx], kept_idx


# =========================================================
# 3. TILE NAME EXTRACTION
# =========================================================
def extract_tile_name(filename):
    """
    Extract tile name from:
    33-2-461-214-22_0_2816.jpg → 33-2-461-214-22
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    return "_".join(parts[:-2])


# =========================================================
# 4. PATCH COUNT ANALYSIS
# =========================================================
def compute_tile_distribution(filenames):
    """
    Count number of patches per tile
    """
    tile_names = [extract_tile_name(f) for f in filenames]
    counts = Counter(tile_names)

    print("Number of tiles:", len(counts))

    values = list(counts.values())
    if values:
        print("Min patches:", min(values))
        print("Max patches:", max(values))
        print("Mean patches:", sum(values) / len(values))

    return counts


# =========================================================
# 5. TILE-BASED SPLIT (NO DATA LEAKAGE)
# =========================================================
def split_by_tiles(
    images,
    masks,
    filenames,
    train_ratio=0.64,
    val_ratio=0.20,
    test_ratio=0.16,
    seed=42
):
    """
    Split dataset by tile names.

    Ensures that all patches from a tile are in the same split.
    """

    if not (len(images) == len(masks) == len(filenames)):
        raise ValueError("images, masks, filenames must have same length")

    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError("Ratios must sum to 1")

    rng = np.random.default_rng(seed)

    filenames = np.array(filenames)
    n_samples = len(filenames)

    # Tile name for each patch
    tile_per_patch = np.array([extract_tile_name(f) for f in filenames])

    # Count patches per tile
    tile_counts = Counter(tile_per_patch)

    # Shuffle + sort tiles by size
    unique_tiles = list(tile_counts.keys())
    rng.shuffle(unique_tiles)
    unique_tiles = sorted(unique_tiles, key=lambda t: tile_counts[t], reverse=True)

    # Targets
    train_target = int(train_ratio * n_samples)
    val_target = int(val_ratio * n_samples)
    test_target = n_samples - train_target - val_target

    split_tiles = {"train": [], "val": [], "test": []}
    split_counts = {"train": 0, "val": 0, "test": 0}
    targets = {"train": train_target, "val": val_target, "test": test_target}

    # Greedy assignment
    for tile in unique_tiles:
        count = tile_counts[tile]

        deficits = {
            split: targets[split] - split_counts[split]
            for split in targets
        }

        chosen_split = max(deficits, key=deficits.get)

        split_tiles[chosen_split].append(tile)
        split_counts[chosen_split] += count

    # Convert to sets
    train_tiles = set(split_tiles["train"])
    val_tiles = set(split_tiles["val"])
    test_tiles = set(split_tiles["test"])

    # Indices
    train_idx = np.where(np.isin(tile_per_patch, list(train_tiles)))[0]
    val_idx = np.where(np.isin(tile_per_patch, list(val_tiles)))[0]
    test_idx = np.where(np.isin(tile_per_patch, list(test_tiles)))[0]

    print("\n=== Split summary ===")
    print(f"Train: {len(train_idx)} ({len(train_idx)/n_samples:.2%})")
    print(f"Val:   {len(val_idx)} ({len(val_idx)/n_samples:.2%})")
    print(f"Test:  {len(test_idx)} ({len(test_idx)/n_samples:.2%})")

    return (
        images[train_idx], images[val_idx], images[test_idx],
        masks[train_idx], masks[val_idx], masks[test_idx],
        filenames[train_idx], filenames[val_idx], filenames[test_idx]
    )


# =========================================================
# 6. FULL PREPROCESS PIPELINE
# =========================================================
def preprocess_dataset(
    images,
    masks,
    filenames,
    empty_keep_ratio=0.25,
    train_ratio=0.64,
    val_ratio=0.20,
    test_ratio=0.16,
    seed=42
):
    """
    Full preprocessing pipeline:
    - normalize images
    - binarize masks
    - filter imbalance
    - split by tiles
    """

    print("\nStep 1 - Normalization")
    images = normalize_image(images)
    masks = binarize_mask(masks)

    print("\nStep 2 - Filtering")
    images, masks, kept_idx = filter_by_mask_content(
        images,
        masks,
        empty_keep_ratio=empty_keep_ratio,
        seed=seed
    )

    filenames = np.array(filenames)[kept_idx]

    print("\nStep 3 - Tile distribution")
    compute_tile_distribution(filenames)

    print("\nStep 4 - Splitting")
    return split_by_tiles(
        images,
        masks,
        filenames,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )