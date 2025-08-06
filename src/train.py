# src/train.py
import sys
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data.dataset import create_segmentation_dataloaders
from src.models.build_model import build_segmentation_model

# Dice Loss for segmentation
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        return 1 - dice

def dice_score(inputs, targets, smooth=1):
    """Calculates the Dice score for validation."""
    inputs = torch.sigmoid(inputs)
    inputs = (inputs > 0.5).float() # Binarize the output
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
    return dice

def run_training(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Training on device: {device} ---")

    train_loader, val_loader = create_segmentation_dataloaders(
        product_name=config['product_name'],
        batch_size=config['batch_size']
    )
    
    model = build_segmentation_model(config['model_config']).to(device)
    
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

    best_val_score = 0.0
    epochs_no_improve = 0
    
    print("--- Starting training loop ---")
    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss_bce = criterion_bce(outputs, masks)
            loss_dice = criterion_dice(outputs, masks)
            loss = loss_bce + loss_dice # Combined loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_dice_score = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_dice_score += dice_score(outputs, masks).item()

        epoch_val_score = val_dice_score / len(val_loader)
        print(f"Epoch {epoch+1}/{config['num_epochs']} | Train Loss: {epoch_loss:.4f} | Val Dice Score: {epoch_val_score:.4f}")

        scheduler.step(epoch_val_score)

        if epoch_val_score > best_val_score:
            print(f"Validation Dice Score improved ({best_val_score:.4f} --> {epoch_val_score:.4f}). Saving model...")
            best_val_score = epoch_val_score
            torch.save(model.state_dict(), config['best_model_path'])
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config['patience']:
            print("Early stopping triggered!")
            break

if __name__ == '__main__':
    TRAINING_CONFIG = {
        'product_name': 'bottle',
        'model_config': {'n_classes': 1, 'n_channels': 3},
        'learning_rate': 1e-4,
        'batch_size': 8,
        'num_epochs': 50,
        'patience': 10,
        'best_model_path': 'models/final/best_segmentation_model.pth'
    }
    run_training(TRAINING_CONFIG)
