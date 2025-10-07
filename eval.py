import torch
import torchvision.transforms as transforms
from model import UNET  # pastikan ini sesuai nama model kamu
from utils import (
    load_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# === KONFIGURASI ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_WORKERS = 2
PIN_MEMORY = True

# direktori dataset (ubah sesuai struktur kamu)
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"

# checkpoint model
CHECKPOINT_PATH = "best_model/UNET-pytorch.pth.tar"

# folder untuk menyimpan hasil prediksi
SAVE_FOLDER = "saved_images_eval/"

# === TRANSFORMASI DATA ===
val_transform = transforms.Compose([
    transforms.ToTensor(),
])

# === INISIALISASI MODEL ===
model = UNET(in_channels=3, out_channels=1).to(DEVICE)

# === LOAD CHECKPOINT ===
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
load_checkpoint(checkpoint, model)

# === DATALOADER VALIDASI ===
_, val_loader = get_loaders(
    train_dir="",  # tidak digunakan untuk eval
    train_maskdir="",
    val_dir=VAL_IMG_DIR,
    val_maskdir=VAL_MASK_DIR,
    batch_size=BATCH_SIZE,
    train_transform=None,  # tidak diperlukan
    val_transform=val_transform,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

# === EVALUASI AKURASI & DICE SCORE ===
check_accuracy(val_loader, model, device=DEVICE)

# === SIMPAN HASIL PREDIKSI ===
save_predictions_as_imgs(val_loader, model, folder=SAVE_FOLDER, device=DEVICE)

print("✅ Evaluasi selesai. Hasil disimpan di:", SAVE_FOLDER)
