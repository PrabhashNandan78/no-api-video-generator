import os
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from modules.pipeline import run_video_factory

if __name__ == "__main__":
    print("Testing full zero-cost AI video factory execution...")
    result = run_video_factory(
        niche="Artificial Intelligence and Robotics",
        geo="IN",
        engine="sdxl",
        background_music=True
    )
    print("\nFACTORY EXECUTION RESULT:")
    for k, v in result.items():
        print(f" {k}: {v}")

