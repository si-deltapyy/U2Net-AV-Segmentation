import os
import torch
import torch.nn as nn
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import U2NET_full  # atau UNet jika kamu ingin uji model lain
from utils import (
    load_checkpoint,
    get_loaders,
    save_predictions_as_imgs,
)

# ================= CONFIG =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_HEIGHT = 2336
IMAGE_WIDTH = 3504
BATCH_SIZE = 1
NUM_WORKERS = 2
PIN_MEMORY = True

VAL_IMG_DIR = "data/HRF/test/images/"
VAL_MASK_DIR = "data/HRF/test/mask/"

CHECKPOINT_PATH = "best_model/best_model.pth.tar"
SAVE_FOLDER = "saved_images_eval/"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# =============== LOSS & METRICS ===============
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
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


# =============== EVALUATION FUNCTION ===============
def evaluate(loader, model, device):
    model.eval()
    dice_total, iou_total, steps = 0.0, 0.0, 0
    dice_fn = DiceLoss()

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Evaluating"):
            imgs = imgs.to(device=device, dtype=torch.float)
            masks = masks.float().unsqueeze(1).to(device=device)

            outputs = model(imgs)

            # U2NET returns list of outputs
            if isinstance(outputs, list) or isinstance(outputs, tuple):
                out = outputs[0]
            else:
                out = outputs

            probs = torch.sigmoid(out)
            dice = 1.0 - dice_fn(out, masks)
            iou = iou_score(probs, masks)

            dice_total += dice.item()
            iou_total += iou
            steps += 1

    avg_dice = dice_total / max(1, steps)
    avg_iou = iou_total / max(1, steps)
    print(f"\n✅ Evaluation Results:")
    print(f"   Dice Score: {avg_dice:.4f}")
    print(f"   IoU Score : {avg_iou:.4f}")
    return avg_dice, avg_iou


# =============== MAIN ===============
def main():
    # Transform validasi (sama seperti di train.py)
    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    print("🚀 Loading model...")
    model = U2NET_full().to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    load_checkpoint(checkpoint, model)

    print("📦 Preparing data loader...")
    _, val_loader = get_loaders(
        train_dir="",
        train_maskdir="",
        val_dir=VAL_IMG_DIR,
        val_maskdir=VAL_MASK_DIR,
        batch_size=BATCH_SIZE,
        train_transform=None,
        val_transform=val_transform,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    print("🔍 Running evaluation...")
    avg_dice, avg_iou = evaluate(val_loader, model, DEVICE)

    print("💾 Saving prediction images...")
    save_predictions_as_imgs(val_loader, model, folder=SAVE_FOLDER, device=DEVICE)

    print("\n✅ Evaluation completed.")
    print(f"Results saved to: {SAVE_FOLDER}")
    print(f"Final Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f}")


if __name__ == "__main__":
    main()
