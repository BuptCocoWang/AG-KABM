import torch
from torch.utils.data import DataLoader, Dataset
import rasterio
import numpy as np
from scipy.interpolate import griddata

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SatelliteDataset(Dataset):
    def __init__(self, file_paths, normalization = True):
        self.file_paths = file_paths
        self.normalization = normalization

    def __len__(self):
        return len(self.file_paths)

    def interpolate_nan_band(self, band):
        
        x = np.arange(0, band.shape[1])
        y = np.arange(0, band.shape[0])
        
        xx, yy = np.meshgrid(x, y)

        valid_points = ~np.isnan(band)
        valid_xx = xx[valid_points]
        valid_yy = yy[valid_points]
        valid_values = band[valid_points]

        filled_band = griddata((valid_xx, valid_yy), valid_values, (xx, yy), method='nearest')

        return filled_band

    def fill_nans(self, image):
        filled_image = np.zeros_like(image)
        for i in range(image.shape[0]):
            filled_image[i] = self.interpolate_nan_band(image[i])

        return filled_image

    def __getitem__(self, idx):
        with rasterio.open(self.file_paths[idx]) as dataset:
            image = dataset.read(out_dtype = "float32")[1:4]


            depth = dataset.read(out_dtype = "float32")[-1]
            
            mask = np.isnan(depth)
            mask = (~mask).astype(float)
            depth[np.isnan(depth)] = 0

            image = self.fill_nans(image)

            image, depth, mask = map(torch.tensor, (image, depth, mask))
            image, depth, mask = image.to(device), depth.to(device), mask.to(device)
            return image.float(), depth.float(), mask.float()