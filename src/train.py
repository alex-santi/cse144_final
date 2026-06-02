import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

SEED = 42


def set_seed(seed: int = 42):
    # Environment variables
    os.environ["PYTHONHASHSEED"] = str(seed)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Library seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CUDA seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # PyTorch backend flags
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        print("Warning: torch.use_deterministic_algorithms(True) not fully supported.")


set_seed(SEED)

from dataset import train_loader, val_loader, num_classes
from model import ResNet18Classifier

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNet18Classifier(num_classes=num_classes).to(device)

# Freeze backbone; train only the new classification head
for param in model.backbone.parameters():
    param.requires_grad = False
for param in model.backbone.fc.parameters():
    param.requires_grad = True

# --- Optional: unfreeze layer4 for fine-tuning (re-enable with optimizer below) ---
# for param in model.backbone.layer4.parameters():
#     param.requires_grad = True

optimizer = torch.optim.Adam(
    model.backbone.fc.parameters(), lr=5e-4, weight_decay=1e-4
)

# --- Optional: differential LR when layer4 is unfrozen ---
# optimizer = torch.optim.Adam([
#     {"params": model.backbone.layer4.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
#     {"params": model.backbone.fc.parameters(), "lr": 1e-3, "weight_decay": 1e-4},
# ])

criterion = nn.CrossEntropyLoss()

# train_one_epoch, evaluate, then the epoch loop with checkpointing
def train_one_epoch(model, loader):
    """Train for one epoch and return (avg_loss, accuracy)."""
    model.train()
    total_loss = 0
    total_accuracy = 0
    total_samples = 0

    for images, labels in loader:
      # use cpu for fixed, reproducible results
      images, labels = images.to(device), labels.to(device)

      # clear feedback
      optimizer.zero_grad()

      prediction = model(images)
      # find the predicted value
      _, predicted_labels = torch.max(prediction, dim=1)
      total_accuracy += (predicted_labels == labels).sum().item()

      loss = criterion(prediction, labels)
      total_loss += loss * images.size(0)

      # updates to be done
      loss.backward()
      # updates weights
      optimizer.step()

      total_samples += labels.size(0)

    ave_loss = total_loss / total_samples
    ave_acc = total_accuracy / total_samples

    return ave_loss.detach(), ave_acc

@torch.no_grad()
def evaluate(model, loader):
    """Evaluate model and return (avg_loss, accuracy)."""
    model.eval()

    total_loss = 0
    total_accuracy = 0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        prediction = model(images)
        _, predicted_labels = torch.max(prediction, dim=1)
        total_accuracy += (predicted_labels == labels).sum().item()

        loss = criterion(prediction, labels)
        total_loss += loss * images.size(0)
        total_samples += labels.size(0)

    ave_loss = total_loss / total_samples
    ave_acc = total_accuracy / total_samples
    return ave_loss.detach(), ave_acc


# Training loop with checkpointing
epochs = 22

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0
best_epoch = -1
ckpt_path = "./checkpoints/best_resnet18.pt"
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

# - Loop through epochs
# - Each epoch: train, validate, log metrics
# - Track best validation accuracy and save checkpoint when improved
# - Print training and validation metrics each epoch
# - For checkpoint: save a dictionary containing keys `model_state_dict` and `epoch`

for t in range(epochs):
  print(f"epoch {t + 1} ---------------------------\n")

  train_ave_loss, train_accuracy = train_one_epoch(model, train_loader)
  print(f"training metrics")
  print(f"average loss: {train_ave_loss}")
  print(f"accuracy: {train_accuracy}\n")

  val_ave_loss, val_accuracy = evaluate(model, val_loader)
  print(f"validation metrics")
  print(f"average loss: {val_ave_loss}")
  print(f"accuracy: {val_accuracy}")

  history["train_loss"].append(train_ave_loss.item())
  history["train_acc"].append(train_accuracy)
  history["val_loss"].append(val_ave_loss.item())
  history["val_acc"].append(val_accuracy)

  if val_accuracy > best_val_acc:
    best_val_acc = val_accuracy
    best_epoch = t + 1

    checkpoint = {"model_state_dict": model.state_dict(), "epoch": best_epoch}

    torch.save(checkpoint, ckpt_path)
    print(f"--> Saved new best checkpoint at epoch {best_epoch}!")

# ========== YOUR CODE ENDS HERE ============

print("Best val acc:", best_val_acc, "at epoch", best_epoch)
print("Saved to:", ckpt_path)

# Plot training curves (overfitting: train improves while val stalls or drops)
epochs_range = range(1, epochs + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(epochs_range, history["train_loss"], "o-", label="train loss")
ax1.plot(epochs_range, history["val_loss"], "s-", label="val loss")
ax1.set_title("Training and Validation Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True)

ax2.plot(epochs_range, [a * 100 for a in history["train_acc"]], "o-", label="train acc")
ax2.plot(epochs_range, [a * 100 for a in history["val_acc"]], "s-", label="val acc")
ax2.axvline(best_epoch, color="gray", linestyle="--", label=f"best val (epoch {best_epoch})")
ax2.set_title("Training and Validation Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("./checkpoints/training_curves.png")
plt.show()




