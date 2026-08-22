import os
import torch
import rasterio
import numpy as np
from skimage.transform import resize

# ==============================================================================
# CONSTANTS & CONFIGURATION
# ==============================================================================

BANDS_ORDER = [
    "B01", "B02", "B03", "B04",
    "B05", "B06", "B07", "B08",
    "B8A", "B09", "B11", "B12"
]

BAND_PATHS = {
    "B01": ("R60m", "B01"),
    "B02": ("R10m", "B02"),
    "B03": ("R10m", "B03"),
    "B04": ("R10m", "B04"),
    "B05": ("R20m", "B05"),
    "B06": ("R20m", "B06"),
    "B07": ("R20m", "B07"),
    "B08": ("R10m", "B08"),
    "B8A": ("R20m", "B8A"),
    "B09": ("R60m", "B09"),
    "B11": ("R20m", "B11"),
    "B12": ("R20m", "B12"),
}

# ==============================================================================
# DATA PROCESSING FUNCTIONS
# ==============================================================================

def upsample_to_10m(data, folder):
    """
    Upsamples lower resolution Sentinel-2 bands to 10m spatial resolution.
    """
    if folder == "R10m":
        return data
    elif folder == "R20m":
        scale = 2
    elif folder == "R60m":
        scale = 6
    else:
        raise ValueError(f"Unknown resolution folder: {folder}")

    h, w = data.shape
    return resize(
        data,
        (h * scale, w * scale),
        preserve_range=True,
        anti_aliasing=True
    )

def extract_patch_dynamic(x0, y0, img_path, size=224):
    """
    Extracts a 12-channel multispectral patch from Sentinel-2 L2A IMG_DATA directory.
    Returns a normalized PyTorch tensor of shape [1, 12, size, size].
    """
    channels = []
    
    for band in BANDS_ORDER:
        folder, _ = BAND_PATHS[band]
        folder_path = os.path.join(img_path, folder)
        
        # Znalezienie odpowiedniego pliku rastrowego .jp2 dla danego kanału
        file = [f for f in os.listdir(folder_path) if f"_{band}_" in f][0]
        path = os.path.join(folder_path, file)

        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)

        # Normalizacja BOA reflectance i usunięcie ewentualnych NaN
        data = data / 10000.0
        data = np.nan_to_num(data)
        
        # Wyrównanie przestrzenne do 10m
        data = upsample_to_10m(data, folder)

        # Wycięcie patcha o odpowiednim rozmiarze
        patch = data[y0:y0+size, x0:x0+size]
        channels.append(patch)

    x = np.stack(channels, axis=0)
    return torch.tensor(x).unsqueeze(0)
