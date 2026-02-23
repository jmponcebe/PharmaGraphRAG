"""Download FDA FAERS quarterly data extract files.

FAERS (FDA Adverse Event Reporting System) provides quarterly CSV exports
of adverse event reports. Each quarter contains several files:
- DEMO: Demographics and administrative info
- DRUG: Drug information per report
- REAC: Adverse reactions per report
- OUTC: Patient outcomes (hospitalization, death, etc.)
- THER: Therapy start/end dates
- INDI: Indications for drug use
- RPSR: Report sources

Usage:
    uv run python -m pharmagraphrag.data.download_faers
    uv run python -m pharmagraphrag.data.download_faers --quarters 2024Q3 2024Q4
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import httpx
from loguru import logger
from tqdm import tqdm

from pharmagraphrag.config import DATA_RAW_DIR, get_settings

# FAERS ASCII data download URL pattern
FAERS_BASE_URL = "https://fis.fda.gov/content/Exports"
FAERS_ZIP_PATTERN = "faers_ascii_{quarter}.zip"

# Files we care about inside the ZIP
FAERS_TABLES = ["DEMO", "DRUG", "REAC", "OUTC", "INDI", "THER", "RPSR"]


def build_download_url(quarter: str) -> str:
    """Build the download URL for a FAERS quarterly ZIP file.

    Args:
        quarter: Quarter string like '2024Q3'.

    Returns:
        Full download URL.
    """
    filename = FAERS_ZIP_PATTERN.format(quarter=quarter)
    return f"{FAERS_BASE_URL}/{filename}"


def download_file(url: str, dest: Path, *, timeout: float = 300.0) -> Path:
    """Download a file with progress bar.

    Args:
        url: URL to download.
        dest: Destination file path.
        timeout: Request timeout in seconds.

    Returns:
        Path to the downloaded file.

    Raises:
        httpx.HTTPStatusError: If the download fails.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        logger.info(f"File already exists, skipping: {dest.name}")
        return dest

    logger.info(f"Downloading {url}")

    with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with (
            open(dest, "wb") as f,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=dest.name,
            ) as progress,
        ):
            for chunk in response.iter_bytes(chunk_size=8192):
                f.write(chunk)
                progress.update(len(chunk))

    logger.success(f"Downloaded: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def extract_zip(zip_path: Path, extract_dir: Path) -> list[Path]:
    """Extract a FAERS ZIP file and return paths to extracted files.

    Args:
        zip_path: Path to the ZIP file.
        extract_dir: Directory to extract to.

    Returns:
        List of extracted file paths.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting {zip_path.name} → {extract_dir}")

    extracted_files = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            # Skip directories and README/deleted files
            if member.endswith("/") or "deleted" in member.lower():
                continue

            # Extract with a flat structure (no nested dirs)
            filename = Path(member).name
            if not filename:
                continue

            target = extract_dir / filename
            if target.exists():
                logger.debug(f"Already extracted: {filename}")
                extracted_files.append(target)
                continue

            with zf.open(member) as source, open(target, "wb") as dest:
                dest.write(source.read())

            extracted_files.append(target)
            logger.debug(f"Extracted: {filename}")

    logger.success(f"Extracted {len(extracted_files)} files to {extract_dir}")
    return extracted_files


def download_quarter(quarter: str, base_dir: Path | None = None) -> Path:
    """Download and extract a single FAERS quarter.

    Args:
        quarter: Quarter string like '2024Q3'.
        base_dir: Base directory for raw data. Defaults to DATA_RAW_DIR.

    Returns:
        Path to the extracted quarter directory.
    """
    base_dir = base_dir or DATA_RAW_DIR
    quarter_dir = base_dir / "faers" / quarter
    zip_dir = base_dir / "faers" / "zips"

    url = build_download_url(quarter)
    zip_path = zip_dir / FAERS_ZIP_PATTERN.format(quarter=quarter)

    # Download
    download_file(url, zip_path)

    # Extract
    extract_zip(zip_path, quarter_dir)

    return quarter_dir


def download_all_quarters(quarters: list[str] | None = None) -> list[Path]:
    """Download and extract all configured FAERS quarters.

    Args:
        quarters: List of quarter strings. Defaults to settings.faers_quarters.

    Returns:
        List of paths to extracted quarter directories.
    """
    settings = get_settings()
    quarters = quarters or settings.faers_quarters

    logger.info(f"Downloading FAERS data for quarters: {quarters}")

    quarter_dirs = []
    for quarter in quarters:
        try:
            quarter_dir = download_quarter(quarter)
            quarter_dirs.append(quarter_dir)
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to download {quarter}: {e}")
            continue

    logger.success(f"Downloaded {len(quarter_dirs)}/{len(quarters)} quarters")
    return quarter_dirs


def main() -> None:
    """CLI entry point for FAERS download."""
    import argparse

    parser = argparse.ArgumentParser(description="Download FDA FAERS quarterly data")
    parser.add_argument(
        "--quarters",
        nargs="+",
        default=None,
        help="Quarters to download (e.g., 2024Q3 2024Q4). Defaults to config.",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    quarter_dirs = download_all_quarters(args.quarters)

    if not quarter_dirs:
        logger.error("No quarters downloaded successfully")
        sys.exit(1)

    for qdir in quarter_dirs:
        files = list(qdir.iterdir())
        logger.info(f"  {qdir.name}: {len(files)} files")
        for f in sorted(files):
            logger.info(f"    - {f.name} ({f.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
