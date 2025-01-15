import os
import argparse
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import r2_score

from AGKABM import AGKABM
from dataset import SatelliteDataset


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the model .pth file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dir = "./test_data"
    test_file_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                      if os.path.isfile(os.path.join(test_dir, f))]
    
    test_dataset = SatelliteDataset(test_file_paths, normalization=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = AGKABM(input_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(args.model_path))

    r2_value, rmse_value = calculate_metrics(model, test_loader, device)
    
    print(f"Test Results:")
    print(f"R²: {r2_value:.4f}")
    print(f"RMSE: {rmse_value:.4f}")


if __name__ == "__main__":
    main()