import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from config import load_config
from captures import schedule_image_capture

def download_fmow(config: dict) -> None:
    data_dir = Path(config.root_dir / config.dataset.root_dir)
    if (data_dir / "rgb_metadata.csv").exists() and \
        (data_dir / "country_code_mapping.csv").exists() and \
        (data_dir / "images").is_dir():
        print(f"[setup_fmow] FMoW dataset already present at {data_dir}, skipped download")
        return
    
    print(f"Downloading FMoW dataset (~54GB), this could take a while...")
    script = Path(config.setup_dir / "download_fmow_dataset.sh")
    subprocess.run(["bash", str(script), str(data_dir)], check=True)

def main():
    example_dir = Path(__file__).parent.parent
    config = load_config(example_dir / "configs/fmow_config.yaml")
    config.root_dir = example_dir
    config.setup_dir = example_dir / "setup"
    download_fmow(config)
    schedule_image_capture(config)

if __name__ == "__main__":
    main()