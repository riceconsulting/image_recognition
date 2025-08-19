# src/train.py
import sys
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

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
    with torch.no_grad():
        inputs = torch.sigmoid(inputs)
        inputs = (inputs > 0.5).float()
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return dice

def run_training(model_architecture: str, dataset_name: str):
    config = get_model_config(model_architecture, dataset_name)
    train_params = config['train_params']
    model_config = config['model_config']
    paths = config['paths']
    
    original_best_path = paths['best_model']
    path_parts = original_best_path.rsplit('.', 1)
    best_model_path = f"{path_parts[0]}_{dataset_name}.{path_parts[1]}"

    original_checkpoint_path = paths['checkpoint']
    path_parts = original_checkpoint_path.rsplit('.', 1)
    checkpoint_path = f"{path_parts[0]}_{dataset_name}.{path_parts[1]}"
    
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
    history = {'train_loss': [], 'val_dice': []}
    
    print("--- Starting training loop ---")
    for epoch in range(train_params['num_epochs']):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, dict): outputs = outputs['out']
            loss = criterion_bce(outputs, masks) + criterion_dice(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)

        model.eval()
        val_dice_score = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                if isinstance(outputs, dict): outputs = outputs['out']
                val_dice_score += dice_score(outputs, masks).item()

        epoch_val_score = val_dice_score / len(val_loader)
        history['val_dice'].append(epoch_val_score)
        print(f"Epoch {epoch+1}/{train_params['num_epochs']} | Loss: {epoch_loss:.4f} | Val Dice: {epoch_val_score:.4f}")

        scheduler.step(epoch_val_score)

        if epoch_val_score > best_val_score:
            print(f"Val Dice improved. Saving model to {best_model_path}")
            best_val_score = epoch_val_score
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= train_params['patience']:
            print("Early stopping triggered!")
            break
            
        torch.save(model.state_dict(), checkpoint_path)
        
    print(f"--- Finished Training. Best model saved to {best_model_path} ---")
    
    history['best_val_score'] = best_val_score
    history['total_epochs'] = epoch + 1
    return history

if __name__ == '__main__':
    from src.config_loader import load_full_config
    full_config = load_full_config()
    active_model = full_config.get('active_model_architecture', 'unet')
    active_dataset = full_config.get('product_name', 'bottle')
    run_training(model_architecture=active_model, dataset_name=active_dataset)
