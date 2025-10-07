# train.py (improved)
import os
import random
from contextlib import nullcontext
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import albumentations as A
from albumentations.pytorch import ToTensorV2

# from model import UNet  # atau U2NET_full() jika ingin pakai U2NET
from model import U2NET_full
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# -------------- HYPERPARAMS / CONFIG ----------------
LEARNING_RATE = 1e-4
BATCH_SIZE = 8                 # kurangi jika OOM; sesuaikan GPU memory
NUM_EPOCHS = 100
NUM_WORKERS = 5
IMAGE_HEIGHT = 2336  # pastikan divisible by 16
IMAGE_WIDTH = 3504   # pastikan divisible by 16
PIN_MEMORY = True
LOAD_MODEL = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_IMG_DIR = "data/HRF/training/images/"
TRAIN_MASK_DIR = "data/HRF/training/mask/"
VAL_IMG_DIR = "data/HRF/test/images/"
VAL_MASK_DIR = "data/HRF/test/mask/"
CHECKPOINT_DIR = "best_model/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
SEED = 42
# ----------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ------- Loss functions: Dice + BCEWithLogits ----------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # preds: logits -> apply sigmoid
        preds = torch.sigmoid(preds)
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (preds * targets).sum(dim=1)
        union = preds.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

def iou_score(preds, targets, threshold=0.5, eps=1e-6):
    preds = (preds > threshold).float()
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()

# ----------------- TRAIN / VALID FUNCTIONS -----------------
def train_fn(loader, model, optimizer, loss_fn, scaler, device, grad_clip=1.0):
    model.train()
    loop = tqdm(loader, desc="Train", leave=False)
    running_loss = 0.0
    steps = 0
    # choose autocast depending on device
    autocast = torch.cuda.amp.autocast if device.type == "cuda" else nullcontext
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device, dtype=torch.float)
        targets = targets.float().unsqueeze(1).to(device=device)

        with autocast():
            outputs = model(data)
            # model may return logits or list (for U2NET). handle both:
            if isinstance(outputs, list) or isinstance(outputs, tuple):
                # primary output assumed to be last fused side (as in U2NET implementation)
                main_out = outputs[0] if len(outputs) == 1 else outputs[0]
                # compute loss as average of side outputs + fused (if present)
                losses = []
                for out in outputs:
                    losses.append(loss_fn(out, targets))
                loss = sum(losses) / len(losses)
            else:
                loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        if device.type == "cuda":
            scaler.scale(loss).backward()
            # gradient clipping (unscale first)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running_loss += loss.item()
        steps += 1
        loop.set_postfix(loss=loss.item())

    return running_loss / max(1, steps)

def eval_fn(loader, model, loss_fn, device):
    model.eval()
    val_loss = 0.0
    steps = 0
    iou_total = 0.0
    dice_total = 0.0
    with torch.no_grad():
        for data, targets in tqdm(loader, desc="Val", leave=False):
            data = data.to(device=device, dtype=torch.float)
            targets = targets.float().unsqueeze(1).to(device=device)

            outputs = model(data)
            if isinstance(outputs, list) or isinstance(outputs, tuple):
                out = outputs[0] if len(outputs) == 1 else outputs[0]
                # if list, usually fused is at index 0 in our modified U2NET; but adapt if needed
            else:
                out = outputs

            loss = loss_fn(out, targets)
            val_loss += loss.item()

            probs = torch.sigmoid(out)
            dice = 1.0 - DiceLoss()(out, targets)  # dice score
            dice_total += dice.item()
            iou_total += iou_score(probs, targets)

            steps += 1

    return val_loss / max(1, steps), dice_total / max(1, steps), iou_total / max(1, steps)

# ----------------- MAIN -----------------
def main():
    # transforms
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            # A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    model = U2NET_full().to(DEVICE)
    # jika ingin pakai U2NET_full: from model import U2NET_full ; model = U2NET_full().to(DEVICE)

    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    # combine BCE + Dice
    def combined_loss(preds, targets, alpha=0.5):
        return alpha * bce(preds, targets) + (1 - alpha) * dice(preds, targets)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY if torch.cuda.is_available() else False,
    )

    if LOAD_MODEL:
        checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pth.tar"), map_location=DEVICE)
        load_checkpoint(checkpoint, model)

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    best_val_iou = 0.0
    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 30)
        train_loss = train_fn(train_loader, model, optimizer, combined_loss, scaler, DEVICE)
        val_loss, val_dice, val_iou = eval_fn(val_loader, model, combined_loss, DEVICE)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}" )
        print(f"Best Val IoU: {best_val_iou:.4f} | Best Val Loss: {best_val_loss:.4f}")

        # scheduler step by val_loss
        scheduler.step(val_loss)

        # save checkpoint (always) and best by IoU
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_iou": val_iou,
        }
        save_checkpoint(checkpoint, filename=os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth.tar"))

        # save best model (by IoU)
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            save_checkpoint(checkpoint, filename=os.path.join(CHECKPOINT_DIR, "best_model.pth.tar"))
            print(f"Saved new best model (IoU={best_val_iou:.4f})")

        # optional: save example predictions
        # save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=DEVICE)

    print("Training finished.")

if __name__ == "__main__":
    main()
