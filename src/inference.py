import csv
import glob
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from dataset import CROP, IMAGENET_MEAN, IMAGENET_STD, RESIZE, idx_to_label, num_classes
from model import ViTB16Classifier
# from model import EfficientNetB3Classifier

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_path = "./checkpoints/best_vit_b16.pt"
submission_path = "./submission.csv"
test_dir = "./data/test"


class TestImageDataset(Dataset):
    def __init__(self, root):
        paths = glob.glob(os.path.join(root, "*.jpg"))
        self.paths = sorted(
            paths, key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        return image, os.path.basename(path)


def normalize_tensor(tensor):
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(3, 1, 1)
    return (tensor - mean) / std


@torch.no_grad()
def predict_ten_crop(model, pil_image):
    """10-crop TTA: FiveCrop + horizontal flip on each crop."""
    resized = TF.resize(pil_image, RESIZE)
    crops = TF.five_crop(resized, CROP)
    logits_sum = None

    for crop in crops:
        for img in (crop, TF.hflip(crop)):
            x = normalize_tensor(TF.to_tensor(img)).unsqueeze(0).to(device)
            logits = model(x)
            logits_sum = logits if logits_sum is None else logits_sum + logits

    return logits_sum.argmax(dim=1).item()


def main():
    dataset = TestImageDataset(test_dir)

    model = ViTB16Classifier(num_classes=num_classes).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    label_map = idx_to_label
    if "class_to_idx" in checkpoint:
        class_to_idx = checkpoint["class_to_idx"]
        label_map = [0] * len(class_to_idx)
        for folder_name, idx in class_to_idx.items():
            label_map[idx] = int(folder_name)

    rows = []
    for image, image_id in dataset:
        pred_idx = predict_ten_crop(model, image)
        rows.append((image_id, label_map[pred_idx]))

    with open(submission_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Label"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} predictions to {submission_path} (10-crop TTA)")
    print("Submit this file to Kaggle for test accuracy.")


if __name__ == "__main__":
    main()
