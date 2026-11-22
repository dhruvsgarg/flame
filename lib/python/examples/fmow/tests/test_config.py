import pytest
from fmow.setup.config import FMoWConfig, load_config

def test_config_defaults():
    config = FMoWConfig()
    assert config.dataset.num_classes == 62
    assert config.dataset.image_size == 224
    assert config.capture.off_nadir_deg == 30.0

def test_config_from_yaml(tmp_path):
    yaml_content = """
    dataset:
        num_classes: 10
        image_size: 128
    capture:
        off_nadir_deg: 45.0
    """

    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml_content)
    config = load_config(config_file)

    assert config.dataset.num_classes == 10
    assert config.dataset.image_size == 128
    assert config.capture.off_nadir_deg == 45.0
    assert config.dataset.root_dir.endswith("lib/python/examples/fmow/data/fmow")


@pytest.mark.parametrize("angle", [-1, 91, float("nan"), float("inf")])
def test_invalid_capture_angle(angle):
    from fmow.setup.config import CaptureConfig
    with pytest.raises(ValueError, match="off_nadir_deg"):
        CaptureConfig(off_nadir_deg=angle)
