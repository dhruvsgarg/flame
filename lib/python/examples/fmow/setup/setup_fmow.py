import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import load_config
from captures import schedule_image_capture

_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "fmow_config.yaml"
_DOWNLOAD_SCRIPT = Path(__file__).parent / "download_fmow_dataset.sh"

def download_fmow(config: dict) -> None:
    data_dir = Path(config.dataset.root_dir)
    if (data_dir / "rgb_metadata.csv").exists() and \
        (data_dir / "country_code_mapping.csv").exists() and \
        (data_dir / "images").is_dir():
        print(f"[setup_fmow] FMoW dataset already present at {data_dir}, skipped download")
        return
    
    print(f"Downloading FMoW dataset (~54GB), this could take a while...")
    subprocess.run(["bash", str(_DOWNLOAD_SCRIPT), str(data_dir)], check=True)

def main():
    config = load_config(_CONFIG_PATH)
    download_fmow(config)
    schedule_image_capture(config)

if __name__ == "__main__":
    main()