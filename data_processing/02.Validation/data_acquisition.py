from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode

import requests


EXPORT_BASE_URL = "https://nvdb-eksport.atlas.vegvesen.no/vegobjekter/915.csv"


def validate_date(date_str: str) -> str:
    """
    Validate that the input date is in YYYY-MM-DD format.
    Returns the same string if valid, raises ValueError otherwise.
    """
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            f"Invalid date '{date_str}'. Expected format: YYYY-MM-DD"
        ) from exc
    return date_str


def build_export_url(
    kommune_code: str,
    date_str: str,
    include: str = "alle",
    extra_params: dict | None = None,
) -> str:
    """
    Build the NVDB Export API URL for vegobjekttype 915 (Vegsystem).
    """
    params = {
        "kommune": kommune_code,
        "tidspunkt": date_str,
        "inkluder": include,
    }

    if extra_params:
        params.update(extra_params)

    return f"{EXPORT_BASE_URL}?{urlencode(params)}"


def is_complete_csv(file_path: Path) -> bool:
    """
    Check whether the downloaded CSV ends with the 'Datauttak komplett' marker.
    This is recommended by NVDB documentation for verifying a complete export.
    """
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return "Datauttak komplett" in content
    except Exception:
        return False


def download_csv(
    kommune_code: str,
    date_str: str,
    output_dir: Path,
    timeout: int = 120,
    include: str = "alle",
    overwrite: bool = False,
    extra_params: dict | None = None,
) -> Path:
    """
    Download one CSV file for one kommune and one date.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"vegsystem_915_kommune_{kommune_code}_{date_str}.csv"
    output_file = output_dir / file_name

    if output_file.exists() and not overwrite:
        print(f"[SKIP] File already exists: {output_file}")
        return output_file

    url = build_export_url(
        kommune_code=kommune_code,
        date_str=date_str,
        include=include,
        extra_params=extra_params,
    )

    print(f"[DOWNLOAD] {url}")

    # A custom User-Agent is a good practice for API-like downloads.
    headers = {
        "User-Agent": "nvdb-vegsystem-downloader/1.0"
    }

    with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
        response.raise_for_status()

        with output_file.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    if is_complete_csv(output_file):
        print(f"[OK] Complete CSV saved to: {output_file}")
    else:
        print(
            f"[WARNING] File saved to {output_file}, "
            "but the completion marker 'Datauttak komplett' was not found. "
            "The export may be incomplete or too large."
        )

    return output_file




