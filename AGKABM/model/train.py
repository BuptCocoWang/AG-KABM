import os
import random
import datetime
from typing import List, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from sklearn.metrics import r2_score

from AGKABM import AGKABM
from dataset import SatelliteDataset
from loss import masked_mse_loss


def seed_torch(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def calculate_metrics(model: torch.nn.Module, 
                     data_loader: DataLoader, 
                     device: torch.device) -> Tuple[float, float]:
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for images, depths, masks in data_loader:
            images = images.to(device)
            outputs = model(images)
            outputs_real = outputs.squeeze().cpu().numpy()
            depths_real = depths.squeeze().cpu().numpy()
            
            valid_indices = masks.squeeze().cpu().numpy().astype(bool)
            predictions.extend(outputs_real[valid_indices])
            targets.extend(depths_real[valid_indices])

    predictions = np.array(predictions)
    targets = np.array(targets)
    r2_value = r2_score(targets, predictions)
    rmse_value = np.sqrt(np.mean((predictions - targets) ** 2))

    return r2_value, rmse_value


def train_model(train_loader: DataLoader,
               val_loader: DataLoader,
               model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               num_epochs: int,
               device: torch.device) -> torch.nn.Module:
    best_r2 = float('-inf')
    best_state_dict = None

    for epoch in range(num_epochs):
        model.train()
        for images, depths, masks in train_loader:
            images, depths, masks = images.to(device), depths.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = masked_mse_loss(outputs, depths, masks)
            loss.backward()
            optimizer.step()

        val_r2, val_rmse = calculate_metrics(model, val_loader, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation R²: {val_r2:.4f}, RMSE: {val_rmse:.4f}')

        if val_r2 > best_r2:
            best_r2 = val_r2
            best_state_dict = model.state_dict().copy()

    return best_state_dict


def main():
    seed_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_dir = "./data/qele"
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                 if os.path.isfile(os.path.join(data_dir, f))]
    
    dataset = SatelliteDataset(file_paths)
    
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    model = AGKABM(input_channels=3, num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_state_dict = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        num_epochs=100,
        device=device
    )
    
    os.makedirs("./pths", exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'./pths/model_{timestamp}.pth'
    torch.save(best_state_dict, save_path)
    print(f'Model saved to {save_path}')


if __name__ == "__main__":
    main()