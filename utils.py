import os
import torch
import torchvision
from dataset import AVDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="best_model/UNET-pytorch.pth.tar"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    print("✅ Saving checkpoint...")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("✅ Loading checkpoint...")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    ):
    train_ds = AVDataset(
        images_dir=train_dir,
        masks_dir=train_maskdir,
        transform=train_transform,
    )

    val_ds = AVDataset(
        images_dir=val_dir,
        masks_dir=val_maskdir,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        persistent_workers=persistent_workers
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        persistent_workers=persistent_workers
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0.0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum().item()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum().item()) / ((preds + y).sum().item() + 1e-8)

    acc = 100 * num_correct / num_pixels
    print(f"Accuracy: {acc:.2f}%")
    print(f"Dice Score: {dice_score / len(loader):.4f}")
    model.train()

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    os.makedirs(folder, exist_ok=True)

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)

        with torch.no_grad():
            outputs = model(x)

            # Jika model mengembalikan beberapa output (seperti U2NET)
            if isinstance(outputs, (list, tuple)):
                preds = torch.sigmoid(outputs[0])
            else:
                preds = torch.sigmoid(outputs)

            preds = (preds > 0.5).float()

        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/gt_{idx}.png")

    model.train()

