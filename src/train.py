# src/train.py
import sys
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

# Add the project root directory to the Python path
# This allows imports from 'src' to work correctly when running this script directly.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Corrected Import: We now import the function that creates the DataLoader directly
from src.data.dataset import get_mvtec_dataloader
from src.models.build_model import build_defect_detection_model

def run_training(config):
    """
    Main training loop.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Training on device: {device} ---")

    # 1. Build Model
    model = build_defect_detection_model(config['model_config']).to(device)

    # 2. Create DataLoader using the new function from dataset.py
    # NOTE 1: Your provided get_mvtec_dataloader function only loads the 'train' dataset.
    # For this reason, the validation data loader and validation logic have been removed from this script.
    # To add validation, you would need to modify get_mvtec_dataloader to also load data from the 'validation' folder.
    print(f"Loading data for category: {config['category']}")
    train_loader = get_mvtec_dataloader(
        category=config['category'],
        batch_size=config['batch_size']
    )

    # NOTE 2: Your dataset.py filters for 'good' samples only. This means the dataloader
    # will only provide one class of data. This will cause an error with CrossEntropyLoss,
    # which expects at least two classes. To fix this for classification, you should remove
    # the line that filters the dataset in dataset.py to ensure both 'good' and defect images are loaded.
    
    # 3. Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'])

    # 4. Training Loop
    print("Starting training...")
    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{config['num_epochs']}], Loss: {running_loss/len(train_loader):.4f}")
        
        # Save a checkpoint
        torch.save(model.state_dict(), config['checkpoint_path'])

    print("--- Finished Training ---")
    
    # 5. Save the final model
    torch.save(model.state_dict(), config['final_model_path'])
    print(f"Final model saved to {config['final_model_path']}")

# --- This block runs when you execute 'python src/train.py' ---
if __name__ == '__main__':
    TRAINING_CONFIG = {
        # The new dataloader function uses a 'category' to build the path
        'category': 'bottle',
        'model_config': {
            'num_classes': 2, # This should match the number of subfolders (e.g., 'good', 'broken') in your data folder
            'pretrained': True
        },
        'learning_rate': 0.001,
        'batch_size': 32,
        'num_epochs': 25,
        'checkpoint_path': 'models/checkpoints/latest.pth',
        'final_model_path': 'models/final/defect_detector_v1.pth'
    }
    run_training(TRAINING_CONFIG)
