import os
import zipfile
import glob
import shutil
import subprocess
from urllib.request import Request, urlopen
from collections import Counter
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_DIR, ORGANISED_DIR, IMG_SIZE, BATCH_SIZE,
    IMAGENET_MEAN, IMAGENET_STD, SEED, NUM_WORKERS
)

HAM10000_FILES = {
    'HAM10000_images_part_1.zip': 'https://dataverse.harvard.edu/api/access/datafile/3172585',
    'HAM10000_images_part_2.zip': 'https://dataverse.harvard.edu/api/access/datafile/3172584',
    'HAM10000_metadata.tab': 'https://dataverse.harvard.edu/api/access/datafile/3172582',
}

def download_public_file(url, dst_path):
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urlopen(req, timeout=180) as resp, open(dst_path, 'wb') as f:
        shutil.copyfileobj(resp, f)

def ensure_ham10000_present():
    current_images = len(glob.glob(os.path.join(DATASET_DIR, '**', '*.jpg'), recursive=True))
    if current_images >= 10000:
        print('Dataset already downloaded and extracted.')
        return

    print('Downloading HAM10000 from Harvard Dataverse...')
    try:
        for fname, url in HAM10000_FILES.items():
            fpath = os.path.join(DATASET_DIR, fname)
            if not os.path.exists(fpath):
                print(f'  -> Downloading {fname}...')
                download_public_file(url, fpath)

        for zip_name in ['HAM10000_images_part_1.zip', 'HAM10000_images_part_2.zip']:
            zip_path = os.path.join(DATASET_DIR, zip_name)
            print(f'  -> Extracting {zip_name}...')
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(DATASET_DIR)
        print('Harvard Dataverse download and extraction complete!')

    except Exception as e:
        print(f'Harvard Dataverse download failed: {e}')
        print('Trying Kaggle API fallback (requires kaggle package and credentials setup)...')
        try:
            subprocess.run([
                'kaggle', 'datasets', 'download', '-d',
                'kmader/skin-cancer-mnist-ham10000',
                '-p', DATASET_DIR, '--unzip', '-q'
            ], check=True)
            print('Kaggle fallback download complete!')
        except Exception as ke:
            print(f'Kaggle fallback failed: {ke}')
            print('Please manually download kmader/skin-cancer-mnist-ham10000 and extract it to:', DATASET_DIR)

def organize_by_class(df):
    """Organizes the images into folders based on their diagnostic category (class)."""
    img_lookup = {}
    for f in glob.glob(os.path.join(DATASET_DIR, '**', '*.jpg'), recursive=True):
        img_lookup[os.path.splitext(os.path.basename(f))[0]] = f

    expected_count = len(df)
    current_count = len(glob.glob(os.path.join(ORGANISED_DIR, '**', '*.jpg'), recursive=True)) if os.path.exists(ORGANISED_DIR) else 0

    if current_count < expected_count:
        if os.path.exists(ORGANISED_DIR):
            shutil.rmtree(ORGANISED_DIR)

        print("Sorting images into class folders based on metadata...")
        for _, row in df.iterrows():
            cls = row['dx']
            img_id = row['image_id']
            dst_dir = os.path.join(ORGANISED_DIR, cls)
            os.makedirs(dst_dir, exist_ok=True)
            src = img_lookup.get(img_id)
            if src:
                dst = os.path.join(dst_dir, os.path.basename(src))
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
        print("Organization complete.")
    else:
        print("Images already organized.")

class SubsetWithTransform(Dataset):
    def __init__(self, base, indices, transform):
        self.base = base
        self.indices = indices
        self.transform = transform
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        path, label = self.base.samples[self.indices[idx]]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

def get_val_test_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

def get_dataset_splits():
    full_dataset = datasets.ImageFolder(root=ORGANISED_DIR)
    class_names = full_dataset.classes
    all_targets = [s[1] for s in full_dataset.samples]
    
    # Train = 75%, Val = 12.5%, Test = 12.5%
    train_idx, temp_idx = train_test_split(
        range(len(full_dataset)), test_size=0.25, random_state=SEED, stratify=all_targets
    )
    temp_targets = [all_targets[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=SEED, stratify=temp_targets
    )
    
    return full_dataset, train_idx, val_idx, test_idx, class_names

def get_dataloaders(full_dataset, train_idx, val_idx, test_idx):
    train_dataset = SubsetWithTransform(full_dataset, train_idx, get_train_transforms())
    val_dataset   = SubsetWithTransform(full_dataset, val_idx,   get_val_test_transforms())
    test_dataset  = SubsetWithTransform(full_dataset, test_idx,  get_val_test_transforms())
    
    # Weighted sampler to address class imbalances
    all_targets = [s[1] for s in full_dataset.samples]
    train_labels = [all_targets[i] for i in train_idx]
    class_sample_counts = Counter(train_labels)
    sample_weights = [1.0 / class_sample_counts[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
                              
    return train_loader, val_loader, test_loader
