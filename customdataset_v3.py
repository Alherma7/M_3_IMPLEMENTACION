import snntorch
from snntorch import spikegen

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import cv2
import numpy as np
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, img_dir, label_dir,timestep=4, edge=True):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.timestep = timestep
        self.edge = edge
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((320, 320))
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.label_dir, self.img_files[idx].replace('.jpg', '.txt'))
        
        image = cv2.imread(img_path)

        if self.edge:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = canny(image)
            
        image = Image.fromarray(image)
        image = self.resize(image)
        image = self.to_tensor(image)
        image = spikegen.rate(image, num_steps=self.timestep)

        labels = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                cls = int(parts[0])
                bbox = list(map(lambda b: float(b), parts[1:5]))
                labels.append([cls] + bbox)

        labels = torch.tensor(labels, dtype=torch.float32)

        return image, labels

def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, 0)  # [B, T, C, H, W]
    formatted = []
    for i, lb in enumerate(labels):
        if lb.shape[0] > 0:
            idx = torch.full((lb.shape[0], 1), i, dtype=torch.float32)
            formatted.append(torch.cat([idx, lb], dim=1))  # [n_obj, 6]
    labels = torch.cat(formatted, 0) if formatted else torch.zeros((0, 6))
    return imgs, labels

def canny(img_gray):
    # Normalizar histograma para estandarizar antes del CLAHE
    img_normalized = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_normalized)
    blurred = cv2.GaussianBlur(img_enhanced, (5, 5), 0)
    median = np.median(blurred)
    lower = int(max(0, 0.2 * median))
    upper = int(min(255, 0.6 * median))
    edges = cv2.Canny(blurred, lower, upper)
    kernel = np.ones((3,3), np.uint8)
    return cv2.dilate(edges, kernel, iterations=1)