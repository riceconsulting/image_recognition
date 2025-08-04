# dataset.py
# Custom PyTorch Dataset class to load MVTec AD data

import os
from torchvision import datasets
from torch.utils.data import DataLoader
from augmentation import get_train_transform  # Make sure both files are in the same folder

def get_mvtec_dataloader(category="bottle", batch_size=32, image_size=256, shuffle=True, num_workers=4):
    # Project root: 3 levels up from src/data/
    base_dir = os.path.normpath(os.path.join(__file__, "..", "..", ".."))
    train_dir = os.path.join(base_dir, "data", "processed", category, "train")

    assert os.path.exists(train_dir), f"Train directory not found: {train_dir}"
    print("Loading from:", train_dir)

    # Load dataset
    transform = get_train_transform(image_size)
    dataset = datasets.ImageFolder(root=train_dir, transform=transform)

    # Optionally filter only 'good' samples
    dataset.samples = [(path, label) for path, label in dataset.samples if "good" in path]

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# Example usage
if __name__ == "__main__":
    loader = get_mvtec_dataloader()
    for images, labels in loader:
        print("Batch:", images.shape)
        break
