# src/data/augmentation.py
import torch
from torchvision import transforms
import random
from PIL import Image

class SegmentationTransform:
    """
    A callable class to apply transformations to an image and its mask for segmentation.
    This structure is compatible with PyTorch's multiprocessing for data loading.
    """
    def __init__(self, image_size=256, is_train=True):
        self.image_size = image_size
        self.is_train = is_train
        self.resize = transforms.Resize(size=(image_size, image_size))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, image, mask):
        # Resize
        image = self.resize(image)
        mask = self.resize(mask)

        # Apply random augmentations only if it's the training set
        if self.is_train:
            # Random horizontal flipping
            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)

            # Random vertical flipping
            if random.random() > 0.5:
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)
            
            # Random rotation
            angle = random.choice([0, 90, 180, 270])
            image = transforms.functional.rotate(image, angle)
            mask = transforms.functional.rotate(mask, angle)

        # Transform to tensor
        image = transforms.functional.to_tensor(image)
        mask = transforms.functional.to_tensor(mask)

        # Normalize image
        image = self.normalize(image)
        
        return image, mask
