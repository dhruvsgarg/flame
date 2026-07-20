from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from model import IMAGENET_MEAN, IMAGENET_STD

class FMoWDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.data = pd.read_csv(self.root_dir / "rgb_metadata.csv")
        self.transform = transform
        self.categories = sorted(self.data["category"].unique())
        self.categories_to_idx = {c: i for i, c in enumerate(self.categories)}

    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        image = Image.open(self.root_dir / "images" / f"rgb_img_{idx}.png").convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.categories_to_idx[row["category"]]
        return image, label

    def get_split_indices(self, split: str) -> np.ndarray:
        return np.flatnonzero((self.data["split"] == split).to_numpy())

    def set_transform(self, image_size: int) -> transforms.Compose:
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

    