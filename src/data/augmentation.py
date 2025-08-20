# src/data/augmentation.py
import torch
from torchvision import transforms
import random
from PIL import Image
import numpy as np

class SegmentationTransform:
    """
    A callable class to apply transformations to an image and its mask for segmentation.
    This structure is compatible with PyTorch's multiprocessing for data loading.
    """
    def __init__(self, image_size=256, is_train=True):
        self.image_size = image_size
        self.is_train = is_train
        
        # --- FIX ---
        # Create separate resize transforms for images (using bilinear) and masks (using nearest).
        # The interpolation mode is set here, during instantiation.
        self.image_resize = transforms.Resize(size=(image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR)
        self.mask_resize = transforms.Resize(size=(image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST)
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, image, mask):
        # --- FIX ---
        # Apply the correct resize transform to each input.
        image = self.image_resize(image)
        mask = self.mask_resize(mask)

        # Apply random augmentations only if it's the training set
        if self.is_train:
            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            if random.random() > 0.5:
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)
            angle = random.choice([0, 90, 180, 270])
            image = transforms.functional.rotate(image, angle)
            mask = transforms.functional.rotate(mask, angle)

        # Convert image to tensor and normalize
        image = transforms.functional.to_tensor(image)
        image = self.normalize(image)
        
        # Convert mask to a LongTensor without adding a channel dimension.
        mask = torch.from_numpy(np.array(mask)).long()
        
        return image, mask
