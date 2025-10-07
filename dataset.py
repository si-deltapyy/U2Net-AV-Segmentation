import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class AVDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = os.listdir(images_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.images_dir, self.images[index])
        image_filename = self.images[index] # Misal isinya "09_g.tif"
        mask_filename = image_filename.replace('_mask.tif', '.jpg') # Hasilnya "09_g.jpg" -> INI SALAH
        mask_path = os.path.join(self.masks_dir, mask_filename)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask