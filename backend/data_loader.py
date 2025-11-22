"""
Data loader that automatically downloads baseline data from GitHub if not available locally.
"""

import os
from pathlib import Path
import urllib.request
import logging

logger = logging.getLogger(__name__)

# GitHub raw content base URL
GITHUB_DATA_URL = "https://raw.githubusercontent.com/GR-Garage-Cup/gr-garage-coach/master/backend/data/models"

def ensure_track_data(track_name: str) -> Path:
    """
    Ensure track baseline data is available locally.
    Downloads from GitHub if not present.

    Args:
        track_name: Name of the track (e.g., 'road_america', 'sonoma')

    Returns:
        Path to the track data directory

    Raises:
        FileNotFoundError: If data cannot be found locally or on GitHub
    """
    # Get backend directory
    backend_dir = Path(__file__).parent
    track_dir = backend_dir / "data" / "models" / track_name

    # Check if all required files exist
    required_files = [
        "champion_baseline.parquet",
        "corners_metadata.parquet",
        "track_segmentation.parquet"
    ]

    all_files_exist = all((track_dir / f).exists() for f in required_files)

    if all_files_exist:
        logger.info(f"Track data for {track_name} found locally")
        return track_dir

    # Create directory if it doesn't exist
    track_dir.mkdir(parents=True, exist_ok=True)

    # Download missing files from GitHub
    logger.info(f"Downloading track data for {track_name} from GitHub...")

    for file_name in required_files:
        file_path = track_dir / file_name

        if file_path.exists():
            logger.debug(f"  {file_name} already exists, skipping")
            continue

        # Download from GitHub
        url = f"{GITHUB_DATA_URL}/{track_name}/{file_name}"

        try:
            logger.info(f"  Downloading {file_name}...")
            urllib.request.urlretrieve(url, file_path)
            logger.info(f"  ✓ Downloaded {file_name}")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.error(f"  ✗ {file_name} not found on GitHub (404)")
                raise FileNotFoundError(
                    f"Track data for '{track_name}' not available. "
                    f"File '{file_name}' not found in GitHub repository."
                )
            else:
                raise

    logger.info(f"✓ Track data for {track_name} ready")
    return track_dir


def list_available_tracks() -> list[str]:
    """
    List all tracks that have data available (locally or on GitHub).

    Returns:
        List of track names
    """
    # For now, return hardcoded list of tracks we know have data
    # This could be enhanced to query GitHub API
    return [
        "barber",
        "cota",
        "indianapolis",
        "road_america",
        "sebring",
        "sonoma",
        "vir"
    ]


if __name__ == "__main__":
    # Test the data loader
    logging.basicConfig(level=logging.INFO)

    print("Testing data loader...")
    print("\nAvailable tracks:", list_available_tracks())

    for track in ["road_america", "sonoma", "barber"]:
        print(f"\n--- Testing {track} ---")
        try:
            track_dir = ensure_track_data(track)
            print(f"✓ Data directory: {track_dir}")

            # List files
            files = list(track_dir.glob("*.parquet"))
            print(f"  Files: {[f.name for f in files]}")
        except Exception as e:
            print(f"✗ Error: {e}")
