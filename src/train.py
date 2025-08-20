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

def mIoU(pred_mask, mask, smooth=1e-10, n_classes=4):
    """
    Calculates mean Intersection over Union for multi-class segmentation.
    """
    with torch.no_grad():
        pred_mask = torch.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(n_classes):
            true_class = pred_mask == clas
            true_label = mask == clas
            if true_label.long().sum().item() == 0: 
                iou_per_class.append(np.nan)
            else:
                intersect = (true_class & true_label).sum().float().item()
                union = (true_class | true_label).sum().float().item()
                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)

def run_training(model_architecture: str, dataset_name: str):
    """
    Main training function that adapts based on the specified model and dataset.
    """
    config = get_model_config(model_architecture, dataset_name)
    train_params = config['train_params']
    model_config = config['model_config']
    paths = config['paths']
    
    best_model_path = f"{paths['best_model'].rsplit('.', 1)[0]}_{dataset_name}.pth"
    checkpoint_path = f"{paths['checkpoint'].rsplit('.', 1)[0]}_{dataset_name}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader = create_segmentation_dataloaders(
        product_name=config['product_name'],
        batch_size=train_params['batch_size']
    )
        
    model = build_model(model_config).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=train_params['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

    best_val_score = 0.0
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_miou': []}
    
    print("--- Starting training loop ---")
    for epoch in range(train_params['num_epochs']):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device).long()
            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, dict): outputs = outputs['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)

        model.eval()
        val_miou_score = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device).long()
                outputs = model(images)
                if isinstance(outputs, dict): outputs = outputs['out']
                val_miou_score += mIoU(outputs, masks, n_classes=model_config['n_classes'])

        epoch_val_score = val_miou_score / len(val_loader)
        history['val_miou'].append(epoch_val_score)
        print(f"Epoch {epoch+1}/{train_params['num_epochs']} | Loss: {epoch_loss:.4f} | Val mIoU: {epoch_val_score:.4f}")

        scheduler.step(epoch_val_score)

        if epoch_val_score > best_val_score:
            print(f"Val mIoU improved. Saving model to {best_model_path}")
            best_val_score = epoch_val_score
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Val mIoU did not improve. Counter: {epochs_no_improve}/{train_params['patience']}")

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
