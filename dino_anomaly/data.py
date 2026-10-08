from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def image_transform(image_size: int) -> v2.Compose:
    return v2.Compose(
        [
            v2.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class MVTecDataset(Dataset):
    """MVTec AD category dataset with image-level labels and pixel masks."""

    def __init__(self, root: str | Path, category: str, split: str, image_size: int):
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        self.category_root = Path(root) / category
        self.split = split
        self.transform = image_transform(image_size)
        self.mask_transform = v2.Compose(
            [
                v2.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.NEAREST
                ),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

        split_root = self.category_root / split
        if not split_root.exists():
            raise FileNotFoundError(f"MVTec split not found: {split_root}")

        extensions = {".png", ".jpg", ".jpeg", ".bmp"}
        candidate_paths = (
            split_root.glob("good/*") if split == "train" else split_root.glob("*/*")
        )
        self.samples = sorted(
            path for path in candidate_paths if path.suffix.lower() in extensions
        )
        if not self.samples:
            raise RuntimeError(f"No images found below {split_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_path = self.samples[index]
        defect_type = image_path.parent.name
        image_tensor = self.transform(Image.open(image_path).convert("RGB"))

        is_anomaly = int(defect_type != "good")
        mask = torch.zeros((1, *image_tensor.shape[-2:]), dtype=torch.float32)
        if self.split == "test" and is_anomaly:
            mask_path = (
                self.category_root
                / "ground_truth"
                / defect_type
                / f"{image_path.stem}_mask.png"
            )
            if not mask_path.exists():
                raise FileNotFoundError(f"Ground-truth mask not found: {mask_path}")
            mask = self.mask_transform(Image.open(mask_path).convert("L"))
            mask = (mask > 0.5).float()

        return {
            "image": image_tensor,
            "mask": mask,
            "label": is_anomaly,
            "defect_type": defect_type,
            "path": str(image_path),
        }
