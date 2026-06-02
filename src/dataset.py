import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

SEED = 42
data_dir = "./data"
batch_size = 64
num_workers = 0

train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

root = f"{data_dir}/train"
full_train_set = ImageFolder(root=root, transform=None)
split_gen = torch.Generator().manual_seed(SEED)
train_idx, val_idx = random_split(full_train_set, [0.8, 0.2], generator=split_gen)

train_set = Subset(ImageFolder(root=root, transform=train_tf), train_idx.indices)
val_set = Subset(ImageFolder(root=root, transform=test_tf), val_idx.indices)

loader_gen = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    generator=loader_gen,
)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=num_workers)

num_classes = len(full_train_set.classes)
