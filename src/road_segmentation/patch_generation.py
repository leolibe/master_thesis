from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
from tqdm import tqdm


def get_patch_positions(width, height, patch_size=512, stride=None, include_borders=True):
    """
    Generate top-left patch positions.

    If include_borders=True, the last patch is forced to touch the right
    and bottom borders so that no area is lost.
    """
    if stride is None:
        stride = patch_size

    if width < patch_size or height < patch_size:
        return [], []

    left_positions = list(range(0, width - patch_size + 1, stride))
    top_positions = list(range(0, height - patch_size + 1, stride))

    if include_borders:
        if left_positions[-1] != width - patch_size:
            left_positions.append(width - patch_size)

        if top_positions[-1] != height - patch_size:
            top_positions.append(height - patch_size)

    return top_positions, left_positions


def save_patch(image_patch, mask_patch, output_image_path, output_mask_path, image_format="jpg", jpeg_quality=95):
    """
    Save one image patch and one mask patch.
    """
    image_patch = image_patch.astype(np.uint8)
    mask_patch = mask_patch.astype(np.uint8)

    if image_format.lower() in ["jpg", "jpeg"]:
        Image.fromarray(image_patch).save(
            output_image_path,
            "JPEG",
            quality=jpeg_quality,
            subsampling=0
        )
    elif image_format.lower() == "png":
        Image.fromarray(image_patch).save(output_image_path)
    else:
        raise ValueError("image_format must be 'jpg', 'jpeg', or 'png'")

    Image.fromarray(mask_patch).save(output_mask_path)


def generate_paired_patches(
    image_dir,
    mask_dir,
    output_dir,
    patch_size=512,
    stride=256,
    min_image_mean=5,
    min_valid_ratio=None,
    empty_keep_ratio=None,
    min_road_ratio=0.0,
    image_format="jpg",
    jpeg_quality=95,
    include_borders=True,
    seed=42
):
    """
    Generate aligned image/mask patches from full tiles.

    Parameters
    ----------
    image_dir : str or Path
        Folder containing full orthophotos (.tif).
    mask_dir : str or Path
        Folder containing full binary masks (.png), one per tile.
    output_dir : str or Path
        Output folder with:
        - images/
        - masks/
    patch_size : int
        Patch size in pixels.
    stride : int
        Sliding-window stride.
    min_image_mean : float
        Skip image patches whose mean pixel value is below this threshold.
        Useful for avoiding black/no-data patches.
    min_valid_ratio : float or None
        If provided, skip patches where the fraction of non-black pixels is below
        this threshold. Example: 0.7.
    empty_keep_ratio : float or None
        If provided, keep all positive patches and only a fraction of empty patches.
        Example: 0.25.
        If None, keep all patches passing the image filters.
    min_road_ratio : float
        Minimum road pixel ratio to consider a patch positive.
    image_format : str
        "jpg" or "png".
    jpeg_quality : int
        JPEG quality when saving images.
    include_borders : bool
        Whether to force border patches.
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)

    out_image_dir = output_dir / "images"
    out_mask_dir = output_dir / "masks"
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(image_dir.glob("*.tif"))

    print(f"TIFF files found: {len(tif_files)}")

    total_saved = 0
    skipped_no_mask = 0
    skipped_shape_mismatch = 0
    skipped_dark = 0
    skipped_invalid = 0
    skipped_empty = 0
    saved_positive = 0
    saved_empty = 0

    for tif_path in tqdm(tif_files, desc="Generating patches"):
        tile_name = tif_path.stem
        mask_path = mask_dir / f"{tile_name}.png"

        if not mask_path.exists():
            skipped_no_mask += 1
            continue

        mask_full = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_full is None:
            skipped_no_mask += 1
            continue

        with rasterio.open(tif_path) as src:
            width, height = src.width, src.height

            if mask_full.shape != (height, width):
                skipped_shape_mismatch += 1
                print(f"[WARNING] Shape mismatch for {tile_name}")
                print(f"Image shape: {(height, width)} | Mask shape: {mask_full.shape}")
                continue

            top_positions, left_positions = get_patch_positions(
                width=width,
                height=height,
                patch_size=patch_size,
                stride=stride,
                include_borders=include_borders
            )

            for top in top_positions:
                for left in left_positions:
                    window = Window(left, top, patch_size, patch_size)

                    image_patch = src.read([1, 2, 3], window=window)
                    image_patch = np.transpose(image_patch, (1, 2, 0))

                    mask_patch = mask_full[
                        top:top + patch_size,
                        left:left + patch_size
                    ]

                    # Skip black / almost empty image patches
                    if image_patch.mean() < min_image_mean:
                        skipped_dark += 1
                        continue

                    # Optional valid-pixel filtering
                    if min_valid_ratio is not None:
                        valid_pixels = np.any(image_patch > 0, axis=2)
                        valid_ratio = valid_pixels.mean()

                        if valid_ratio < min_valid_ratio:
                            skipped_invalid += 1
                            continue

                    mask_binary = (mask_patch > 0).astype(np.uint8)
                    road_ratio = mask_binary.mean()

                    # Optional empty patch filtering
                    if empty_keep_ratio is not None:
                        if road_ratio > min_road_ratio:
                            keep = True
                            saved_positive += 1
                        else:
                            keep = rng.random() < empty_keep_ratio
                            if keep:
                                saved_empty += 1
                            else:
                                skipped_empty += 1

                        if not keep:
                            continue

                    filename = f"{tile_name}_{top}_{left}"

                    image_save_path = out_image_dir / f"{filename}.{image_format}"
                    mask_save_path = out_mask_dir / f"{filename}.png"

                    # Save mask as 0/255
                    mask_to_save = mask_binary * 255

                    save_patch(
                        image_patch=image_patch,
                        mask_patch=mask_to_save,
                        output_image_path=image_save_path,
                        output_mask_path=mask_save_path,
                        image_format=image_format,
                        jpeg_quality=jpeg_quality
                    )

                    total_saved += 1

                    if empty_keep_ratio is None:
                        if road_ratio > min_road_ratio:
                            saved_positive += 1
                        else:
                            saved_empty += 1

    print("\n=== Patch generation summary ===")
    print(f"Saved patches:              {total_saved}")
    print(f"Saved positive patches:     {saved_positive}")
    print(f"Saved empty patches:        {saved_empty}")
    print(f"Skipped missing masks:      {skipped_no_mask}")
    print(f"Skipped shape mismatch:     {skipped_shape_mismatch}")
    print(f"Skipped dark patches:       {skipped_dark}")
    print(f"Skipped invalid patches:    {skipped_invalid}")
    print(f"Skipped empty patches:      {skipped_empty}")


def analyze_patch_dataset(mask_dir):
    """
    Analyze a mask patch dataset.
    """
    mask_dir = Path(mask_dir)
    mask_files = sorted(mask_dir.glob("*.png"))

    positive = 0
    empty = 0
    road_ratios = []

    for mask_path in tqdm(mask_files, desc="Analyzing masks"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        mask_binary = (mask > 0).astype(np.uint8)
        road_ratio = mask_binary.mean()
        road_ratios.append(road_ratio)

        if road_ratio > 0:
            positive += 1
        else:
            empty += 1

    total = positive + empty

    print("\n=== Patch dataset analysis ===")
    print(f"Total patches:    {total}")
    print(f"Positive patches: {positive}")
    print(f"Empty patches:    {empty}")

    if total > 0:
        print(f"Positive ratio:   {positive / total:.3f}")
        print(f"Empty ratio:      {empty / total:.3f}")

    if road_ratios:
        print(f"Mean road ratio:  {np.mean(road_ratios):.6f}")
        print(f"Max road ratio:   {np.max(road_ratios):.6f}")