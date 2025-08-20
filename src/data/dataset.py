# src/data/dataset.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from .augmentation import SegmentationTransform
import json

class SegmentationDataset(Dataset):
    def __init__(self, data_dir, class_map, transform=None):
        self.transform = transform
        self.class_map = class_map
        self.images = []
        self.masks = []

        image_folders = [os.path.join(data_dir, 'train'), os.path.join(data_dir, 'test')]
        for folder in image_folders:
            if not os.path.isdir(folder): continue
            
            for defect_type in os.listdir(folder):
                defect_folder = os.path.join(folder, defect_type)
                if not os.path.isdir(defect_folder): continue

                for img_name in os.listdir(defect_folder):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(defect_folder, img_name)
                        self.images.append(img_path)

                        mask_path = None
                        if defect_type != 'good':
                            # Load from the new multi-class mask directory
                            mask_folder = os.path.join(data_dir, 'ground_truth_multiclass', defect_type)
                            mask_name = os.path.splitext(img_name)[0] + '_mask.png'
                            potential_mask_path = os.path.join(mask_folder, mask_name)
                            if os.path.exists(potential_mask_path):
                                mask_path = potential_mask_path
                        
                        self.masks.append(mask_path)

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_path, mask_path = self.images[idx], self.masks[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path) if mask_path else Image.new('L', image.size, 0)

        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask

def create_segmentation_dataloaders(product_name, batch_size=32, image_size=256, val_split=0.2, num_workers=4):
    base_dir = os.path.normpath(os.path.join(__file__, "..", "..", ".."))
    processed_dir = os.path.join(base_dir, "data", "processed", product_name)
    
    # Load the class_map from the dataset's labels.json
    labels_path = os.path.join(processed_dir, 'labels.json')
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels.json not found for '{product_name}'. Please run preprocess_masks.py first.")
    with open(labels_path, 'r') as f:
        class_map = json.load(f)

    train_dataset = SegmentationDataset(processed_dir, class_map, transform=SegmentationTransform(image_size, is_train=True))
    val_dataset = SegmentationDataset(processed_dir, class_map, transform=SegmentationTransform(image_size, is_train=False))

    indices = list(range(len(train_dataset)))
    val_size = int(len(indices) * val_split)
    train_size = len(indices) - val_size
    train_indices, val_indices = random_split(indices, [train_size, val_size])

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    if os.name == 'nt': num_workers = 0

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"Data split into {len(train_subset)} training samples and {len(val_subset)} validation samples.")
    return train_loader, validation_loader
