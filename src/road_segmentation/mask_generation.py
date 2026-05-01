from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely import wkb
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd


# =========================================================
# 1. LOAD AND PREPARE GML
# =========================================================
def load_and_prepare_gml(gml_path, layer=None, target_crs="EPSG:25833"):
    """
    Load GML file and prepare geometries:
    - select layer
    - convert CRS
    - remove Z dimension
    """

    gdf = gpd.read_file(gml_path, layer=layer)

    # Convert CRS
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    # Remove Z dimension
    gdf["geometry"] = gdf["geometry"].apply(
        lambda geom: wkb.loads(wkb.dumps(geom, output_dimension=2))
    )

    return gdf


# =========================================================
# 2. MERGE MULTIPLE GDFs
# =========================================================
def merge_gdfs(gdf_list):
    """
    Merge multiple GeoDataFrames into one
    """
    merged = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
    merged.crs = gdf_list[0].crs
    return merged


# =========================================================
# 3. RASTERIZE FOR ONE TILE
# =========================================================
def rasterize_roads_for_tile(roads_gdf, tif_path):
    """
    Rasterize road geometries aligned with a GeoTIFF
    """

    with rasterio.open(tif_path) as src:
        transform = src.transform
        height = src.height
        width = src.width
        bounds = src.bounds

    # Clip geometries to tile extent (important for performance)
    roads_clipped = roads_gdf.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]

    shapes = [(geom, 1) for geom in roads_clipped.geometry if geom is not None]

    if len(shapes) == 0:
        mask = np.zeros((height, width), dtype=np.uint8)
    else:
        mask = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

    return mask


# =========================================================
# 4. GENERATE MASKS FOR ALL TILES
# =========================================================
def generate_masks_from_gml(
    tif_dir,
    gml_path,
    output_dir,
    layer=None,
    target_crs="EPSG:25833"
):
    """
    Main pipeline:
    - load GML
    - rasterize for each tif
    - save masks as PNG
    """

    tif_dir = Path(tif_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(tif_dir.glob("*.tif"))
    print(f"{len(tif_files)} TIFF files found")

    # Load roads
    print("Loading GML...")
    roads_gdf = load_and_prepare_gml(gml_path, layer, target_crs)

    print("Rasterizing masks...")
    for tif_path in tqdm(tif_files):
        mask = rasterize_roads_for_tile(roads_gdf, tif_path)

        output_path = output_dir / f"{tif_path.stem}.png"

        Image.fromarray(mask * 255).save(output_path)

    print("✅ Mask generation complete")