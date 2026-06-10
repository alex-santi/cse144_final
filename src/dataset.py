import os

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


class NumericImageFolder(ImageFolder):
    """Sort classes 0,1,...,99 so that the label index matches folder name"""

    @staticmethod
    def find_classes(directory):
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        classes.sort(key=int)
        class_to_idx = {name: int(name) for name in classes}
        return classes, class_to_idx

# keep it consistent across runs
SEED = 42
data_dir = "./data"
batch_size = 32
num_workers = 0 # For fully reproducible ordering across platforms
val_ratio = 0.2 # used for train/val split

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ViT-B/16
# - resize 256, crop 224
# EfficientNet-B3
#  - Resize(320), crop 300
RESIZE = 256
CROP = 224

train_tf = transforms.Compose([
    transforms.Resize(RESIZE),
    transforms.RandomResizedCrop(CROP, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.3),
])

val_tf = transforms.Compose([
    transforms.Resize(RESIZE),
    transforms.CenterCrop(CROP),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

test_tf = val_tf

_train_root = f"{data_dir}/train"
train_set = NumericImageFolder(root=_train_root, transform=train_tf)

loader_gen = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    generator=loader_gen,
)

# # --- 80/20 train/val split ---
# _train_indexed = NumericImageFolder(root=_train_root, transform=train_tf)
# _val_indexed = NumericImageFolder(root=_train_root, transform=val_tf)
# n_total = len(_train_indexed)
# n_val = int(val_ratio * n_total)
# n_train = n_total - n_val
# split_gen = torch.Generator().manual_seed(SEED)
# train_set, val_subset = random_split(
#     _train_indexed, [n_train, n_val], generator=split_gen
# )
# val_set = Subset(_val_indexed, val_subset.indices)
# train_loader = DataLoader(
#     train_set, batch_size=batch_size, shuffle=True,
#     num_workers=num_workers, generator=loader_gen,
# )
# val_loader = DataLoader(
#     val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
# )


# # --- 80/20 train/val split ---
# num_classes = len(_train_indexed.classes)
# class_to_idx = _train_indexed.class_to_idx
# idx_to_label = [int(name) for name in _train_indexed.classes]


#### replace ####
num_classes = len(train_set.classes)
class_to_idx = train_set.class_to_idx
idx_to_label = [int(name) for name in train_set.classes]
#### replace ####

# # --- 80/20 train/val split ---
# print(f"Train: {len(train_set)} images ({100 * (1 - val_ratio):.0f}% of data/train/)")
# print(f"Val:   {len(val_set)} images ({100 * val_ratio:.0f}% of data/train/)")

#### replace ####
print(f"Train: {len(train_set)} images (full data/train/)")
#### replace ####

print("Kaggle test: unlabeled images in data/test/ — submit submission.csv")
