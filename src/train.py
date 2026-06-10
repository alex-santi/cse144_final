import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

SEED = 42
BASE_LR = 3e-3
# WEIGHT_DECAY = 2e-4
MIXUP_ALPHA = 0.2
GRAD_CLIP = 1.0

# 
def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        print("Warning: torch.use_deterministic_algorithms(True) not fully supported.")


set_seed(SEED)

from dataset import class_to_idx, train_loader, num_classes
# from dataset import val_loader  # uncomment with train/val split in dataset.py
from model import ViTB16Classifier
# from model import EfficientNetB3Classifier

# For fixed, reproducible results. (You may switch to "cuda" after you finish debugging.)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ViTB16Classifier(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
ckpt_path = "./checkpoints/best_vit_b16.pt"
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

PHASE_A_EPOCHS = 15
PHASE_B_EPOCHS = 15
PHASE_B_BLOCK_LR = 3e-5
PHASE_B_HEAD_LR = 1e-4
# PHASE_A_EARLY_STOP = 4


def mixup_batch(images, labels):
    lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
    index = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[index]
    return mixed, labels, labels[index], lam


def train_one_epoch(model, loader, optimizer):
    model.train()

    total_loss = 0.0
    total_accuracy = 0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        images, labels_a, labels_b, lam = mixup_batch(images, labels)
        optimizer.zero_grad()

        prediction = model(images)
        loss = lam * criterion(prediction, labels_a) + (1 - lam) * criterion(prediction, labels_b)

        _, predicted_labels = torch.max(prediction, dim=1)
        batch_acc = (
            lam * (predicted_labels == labels_a).float()
            + (1 - lam) * (predicted_labels == labels_b).float()
        ).sum().item()

        total_accuracy += batch_acc
        total_loss += loss.item() * images.size(0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_samples += labels.size(0)

    return total_loss / total_samples, total_accuracy / total_samples


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        prediction = model(images)
        _, predicted_labels = torch.max(prediction, dim=1)
        total_accuracy += (predicted_labels == labels).sum().item()
        loss = criterion(prediction, labels)
        total_loss += loss.item() * images.size(0)
        total_samples += labels.size(0)

    return total_loss / total_samples, total_accuracy / total_samples


# # --- 80/20 train/val split ---
# history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
# best_val_acc = 0.0
history = {"train_loss": [], "train_acc": []} 
best_train_acc = 0.0
best_epoch = -1


def run_training(epochs, phase_name, save_path, optimizer, early_stop_patience=None):
    global best_train_acc, best_epoch

    for t in range(epochs):
        print(f"{phase_name} epoch {t + 1} ---------------------------\n")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        # val_loss, val_acc = evaluate(model, val_loader)

        print("training metrics")
        print(f"average loss: {train_loss:.4f}")
        print(f"accuracy: {train_acc:.4f}\n")
        # print("validation metrics")
        # print(f"average loss: {val_loss:.4f}")
        # print(f"accuracy: {val_acc:.4f}\n")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        # history["val_loss"].append(val_loss)
        # history["val_acc"].append(val_acc)


        # # --- 80/20 train/val split ---
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     best_epoch = len(history["val_acc"])
        #     epochs_without_improvement = 0

        if train_acc > best_train_acc: # replace
            best_train_acc = train_acc
            best_epoch = len(history["train_acc"]) # replace
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": best_epoch,
                    "phase": phase_name,
                    # # --- 80/20 train/val split ---
                    # "val_acc": val_acc, 
                    "train_acc": train_acc, # replace
                    "class_to_idx": class_to_idx,
                },
                save_path,
            )
        
        # # --- 80/20 train/val split ---
        #     print(f"--> Saved new best checkpoint (val acc {val_acc:.4f}) at {phase_name} epoch {t + 1}!")
        # elif early_stop_patience is not None:
        #     epochs_without_improvement += 1
        #     if epochs_without_improvement >= early_stop_patience:
        #         print(
        #             f"Early stopping {phase_name}: no val improvement for "
        #             f"{early_stop_patience} epochs (best val {best_val_acc:.4f}).\n"
        #         )
        #         break

        # for kaggle eval
        print(f"--> Saved new best checkpoint (train acc {train_acc:.4f}) at {phase_name} epoch {t + 1}!")

        print()


print("========== Phase A: train classification head only ==========\n")
for param in model.backbone.parameters():
    param.requires_grad = False
for param in model.backbone.fc.parameters():
    param.requires_grad = True

# TESTING with weight decay
# optimizer_a = torch.optim.AdamW(
#     model.backbone.fc.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY
# )

optimizer_a = torch.optim.AdamW(model.backbone.fc.parameters(), lr=BASE_LR)

run_training(PHASE_A_EPOCHS, "Phase A (head only)", ckpt_path, optimizer_a)

# Phase B: fine-tune last block + head (load Phase A best)
print("========== Phase B: fine-tune last block + head ==========\n")
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
print(
    f"Loaded Phase A best from epoch {checkpoint['epoch']} "
    # # --- 80/20 train/val split ---
    # f"(val acc {checkpoint.get('val_acc', 'n/a')})\n"

    
    f"(train acc {checkpoint.get('train_acc', 'n/a')})\n" # replace
)

for param in model.backbone.parameters():
    param.requires_grad = False
for param in model.backbone.layer4.parameters():
    param.requires_grad = True
for param in model.backbone.fc.parameters():
    param.requires_grad = True

optimizer_b = torch.optim.AdamW([
    {"params": model.backbone.layer4.parameters(), "lr": PHASE_B_BLOCK_LR},
    {"params": model.backbone.fc.parameters(), "lr": PHASE_B_HEAD_LR},
])
run_training(PHASE_B_EPOCHS, "Phase B (block + head)", ckpt_path, optimizer_b)

best_ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(best_ckpt["model_state_dict"])
# # --- 80/20 train/val split ---
# print("Reloaded best val checkpoint into model.")

print("Reloaded best train checkpoint into model.") # replace

# # --- 80/20 train/val split ---
# print("Best val acc:", best_val_acc, "at epoch", best_epoch)
print("Best train acc:", best_train_acc, "at epoch", best_epoch) # reaplce

print("Best checkpoint (submit via inference):", ckpt_path)
print("Inference: PYTHONPATH=src python src/inference.py")

epochs_range = range(1, len(history["train_loss"]) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(epochs_range, history["train_loss"], "o-", label="train loss")
# ax1.plot(epochs_range, history["val_loss"], "s-", label="val loss")
ax1.set_title("Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True)

ax2.plot(epochs_range, [a * 100 for a in history["train_acc"]], "o-", label="train acc")
# ax2.plot(epochs_range, [a * 100 for a in history["val_acc"]], "s-", label="val acc")

# # --- 80/20 train/val split ---
# ax2.axvline(best_epoch, color="gray", linestyle="--", label=f"best val (epoch {best_epoch})")

ax2.axvline(best_epoch, color="gray", linestyle="--", label=f"best checkpoint (epoch {best_epoch})") # replace
ax2.set_title("Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("./checkpoints/training_curves.png")
print("Saved plot: ./checkpoints/training_curves.png")
plt.show()




