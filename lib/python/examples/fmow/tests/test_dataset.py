import pytest
import pandas as pd
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from fmow.dependencies.fmow_dataset import FMoWDataset

@pytest.fixture
def dummy_dataset(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img0 = Image.new("RGB", (100, 100), color="red")
    img0.save(images_dir / "rgb_img_0.png")

    img1 = Image.new("RGB", (100, 100), color="blue")
    img1.save(images_dir / "rgb_img_1.png")

    dummy_csv = tmp_path / "rgb_metadata.csv"
    dummy_data = pd.DataFrame({
        "category": ["airport", "airport_hangar"],
        "split": ["train", "val"]
    })
    dummy_data.to_csv(dummy_csv, index=False)

    return FMoWDataset(root_dir=tmp_path)



# Test if FMoW Dataset correctly initializes
def test_fmow_dataset_loading(dummy_dataset):
    assert len(dummy_dataset) == 2
    assert dummy_dataset.categories_to_idx["airport"] == 0
    assert dummy_dataset.categories_to_idx["airport_hangar"] == 1


# Tests if teh dataset correctly identifies all the indices in a split
def test_fmow_dataset_split_indices(dummy_dataset):
    train_indices = dummy_dataset.get_split_indices("train")
    assert len(train_indices) == 1
    assert train_indices[0] == 0

# Tests if the dataset correctly gets a non-transformed image index
def test_fmow_dataset_getitem_no_transform(dummy_dataset):
    image, label = dummy_dataset[0]
    assert isinstance(image, Image.Image)
    assert label == 0

# Tests if the dataset correctly gets a transformed image index and that the transform was successful
def test_fmow_dataset_getitem_with_transform(dummy_dataset):
    dummy_dataset.set_transform(image_size=224)
    image, label = dummy_dataset[1]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert label == 1
