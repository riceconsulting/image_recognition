# src/evaluate.py
import sys
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.models.build_model import build_segmentation_model
from src.data.dataset import SegmentationDataset # Import the custom dataset
from src.train import dice_score # Import dice_score from train script

def evaluate_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Evaluation on device: {device} ---")

    # Use the same dataset logic but point to the test set
    # For MVTec, the test set is what we evaluate on.
    test_dir = os.path.join(project_root, "data", "processed", config['product_name'])
    test_dataset = SegmentationDataset(test_dir) # No augmentations for testing
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    model = build_segmentation_model(config['model_config'])
    model.load_state_dict(torch.load(config['best_model_path'], map_location=device))
    model.to(device)
    model.eval()

    total_dice_score = 0
    
    # Create a directory to save visualization results
    output_dir = os.path.join(project_root, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            total_dice_score += dice_score(outputs, masks).item()
            
            # Save the first few results for visualization
            if i < config['num_visualizations']:
                # Process for visualization
                img_np = images[0].cpu().numpy().transpose(1, 2, 0)
                gt_mask_np = masks[0].cpu().numpy().squeeze()
                pred_mask_np = (torch.sigmoid(outputs[0]) > 0.5).cpu().numpy().squeeze()

                # Un-normalize the image for display
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = std * img_np + mean
                img_np = np.clip(img_np, 0, 1)

                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(img_np)
                ax[0].set_title("Input Image")
                ax[0].axis('off')
                
                ax[1].imshow(gt_mask_np, cmap='gray')
                ax[1].set_title("Ground Truth Mask")
                ax[1].axis('off')

                ax[2].imshow(pred_mask_np, cmap='gray')
                ax[2].set_title("Predicted Mask")
                ax[2].axis('off')
                
                plt.savefig(os.path.join(output_dir, f"result_{i}.png"))
                plt.close()

    avg_dice_score = total_dice_score / len(test_loader)
    print(f"\n--- Evaluation Complete ---")
    print(f"Average Dice Score on Test Set: {avg_dice_score:.4f}")
    print(f"Visualizations saved to: {output_dir}")

if __name__ == '__main__':
    EVAL_CONFIG = {
        'product_name': 'bottle',
        'model_config': {'n_classes': 1, 'n_channels': 3},
        'batch_size': 1, # Evaluate one image at a time for visualization
        'best_model_path': 'models/final/best_segmentation_model.pth',
        'num_visualizations': 10 # Number of example images to save
    }
    evaluate_model(EVAL_CONFIG)
