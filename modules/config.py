"""
Configuration and Path Management for Local Video Factory.
Works seamlessly both inside Google Colab and in standalone environments.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

# Ensure standard output can handle utf-8 characters without cp1252 exceptions
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Setup standard logging
LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("VideoFactory")

def get_base_directories(force_local: bool = False):
    """
    Determines and returns the base directory structure.
    Checks if Google Drive is mounted at /content/drive/MyDrive/.
    If available, outputs are saved there for persistence across Colab restarts.
    Otherwise, uses a local fallback directory.
    """
    in_colab = "google.colab" in sys.modules or os.path.exists("/content")
    
    if in_colab and not force_local:
        drive_path = Path("/content/drive/MyDrive/VideoFactory")
        if Path("/content/drive/MyDrive").exists():
            base_dir = drive_path
            logger.info(f"Google Drive mounted. Persisting output to: {base_dir}")
        else:
            base_dir = Path("/content/VideoFactory")
            logger.warning(f"Google Drive not mounted. Using local storage: {base_dir}")
        temp_dir = Path("/content/temp/VideoFactory")
    else:
        # Local execution / development mode
        script_dir = Path(__file__).resolve().parent.parent
        base_dir = script_dir / "VideoFactory"
        temp_dir = script_dir / "temp"

    # Define subdirectories
    dirs = {
        "base": base_dir,
        "outputs": base_dir / "outputs",
        "scenes": base_dir / "scenes",
        "audio": base_dir / "audio",
        "images": base_dir / "images",
        "scripts": base_dir / "scripts",
        "logs": base_dir / "logs",
        "temp": temp_dir,
    }

    # Ensure all directories exist
    for name, p in dirs.items():
        p.mkdir(parents=True, exist_ok=True)

    # Configure file logging
    log_file = dirs["logs"] / "factory.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)

    return dirs
