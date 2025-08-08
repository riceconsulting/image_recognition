# src/train.py
import sys
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.config_loader import get_model_config
from src.data.dataset import create_segmentation_dataloaders
from src.models.build_model import build_model

# --- Loss Functions & Metrics ---
class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

def dice_score(inputs, targets, smooth=1):
    """Calculates the Dice score for validation."""
    with torch.no_grad():
        inputs = torch.sigmoid(inputs)
        inputs = (inputs > 0.5).float()  # Binarize the output
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return dice

def run_training(model_architecture: str):
    """
    Main training function that adapts based on the specified model architecture.
    """
    config = get_model_config(model_architecture)
    train_params = config['train_params']
    model_config = config['model_config']
    paths = config['paths']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Training on device: {device} ---")

    train_loader, val_loader = create_segmentation_dataloaders(
        product_name=config['product_name'],
        batch_size=train_params['batch_size']
    )
        
    model = build_model(model_config).to(device)
    
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=train_params['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

    best_val_score = 0.0
    epochs_no_improve = 0
    
    print("--- Starting training loop ---")
    for epoch in range(train_params['num_epochs']):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            # Handle DeepLabV3+ output which is a dict
            if isinstance(outputs, dict):
                outputs = outputs['out']
            loss_bce = criterion_bce(outputs, masks)
            loss_dice = criterion_dice(outputs, masks)
            loss = loss_bce + loss_dice # Combined loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        # --- FIX ---
        # Get the length of the subset, not the original dataset's non-existent attribute
        epoch_loss = running_loss / len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_dice_score = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                if isinstance(outputs, dict):
                    outputs = outputs['out']
                val_dice_score += dice_score(outputs, masks).item()

        epoch_val_score = val_dice_score / len(val_loader)
        print(f"Epoch {epoch+1}/{train_params['num_epochs']} | Train Loss: {epoch_loss:.4f} | Val Dice Score: {epoch_val_score:.4f}")

        scheduler.step(epoch_val_score)

        # --- Early Stopping and Model Saving ---
        if epoch_val_score > best_val_score:
            print(f"Validation Dice Score improved ({best_val_score:.4f} --> {epoch_val_score:.4f}). Saving model...")
            best_val_score = epoch_val_score
            torch.save(model.state_dict(), paths['best_model'])
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= train_params['patience']:
            print("Early stopping triggered!")
            break
            
        # Also save the latest checkpoint
        torch.save(model.state_dict(), paths['checkpoint'])
        
    print("--- Finished Training ---")
    print(f"Best model saved to {paths['best_model']}")

if __name__ == '__main__':
    # Example of how to run training for a specific model directly
    # To change the model, you would change the argument here or in the config.yaml
    from src.config_loader import load_full_config
    active_model = load_full_config().get('active_model_architecture', 'unet')
    run_training(model_architecture=active_model)
