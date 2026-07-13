from dataclasses import dataclass, field
from pathlib import Path
import yaml

@dataclass
class DatasetConfig:
    root_dir: str = "data/fmow"
    num_classes: int = 62
    image_size: int = 224

@dataclass
class SatellitesConfig:
    leo_dir: str = "metadata/leo"

@dataclass
class CaptureConfig:
    radius: float = 15.0

@dataclass
class FMoWConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    satellites: SatellitesConfig = field(default_factory=SatellitesConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FMoWConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        
        dataset_data = data.get("dataset", {})
        satellites_data = data.get("satellites", {})
        capture_data = data.get("capture", {})

        return cls(
            dataset=DatasetConfig(**dataset_data)if dataset_data else DatasetConfig(),
            satellites=SatellitesConfig(**satellites_data) if satellites_data else SatellitesConfig(),
            capture=CaptureConfig(**capture_data) if capture_data else CaptureConfig()
        )

def load_config(path: str | Path) -> FMoWConfig:
    return FMoWConfig.from_yaml(path)