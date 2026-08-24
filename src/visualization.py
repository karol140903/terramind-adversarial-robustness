import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ==============================================================================
# INDICES CALCULATION HELPERS
# ==============================================================================

def calculate_indices(tensor):
    """
    Calculates RGB and spectral indices (NDVI, NDWI, NDBI) from a 12-channel Sentinel-2 tensor.
    Expected band indices: Green=2, Red=3, NIR=7, SWIR=10
    """
    epsilon = 1e-8
    green = tensor[2]
    red = tensor[3]
    nir = tensor[7]
    swir = tensor[10]

    ndvi = (nir - red) / (nir + red + epsilon)
    ndwi = (green - nir) / (green + nir + epsilon)
    ndbi = (swir - nir) / (swir + nir + epsilon)

    rgb = np.stack([red, green, tensor[1]], axis=-1)
    p99 = np.percentile(rgb, 99)
    rgb = np.clip(rgb / (p99 + epsilon), 0, 1)

    return rgb, ndvi, ndwi, ndbi

# ==============================================================================
# RGB & SINGLE PATCH VISUALIZATIONS
# ==============================================================================

def plot_rgb_triplet(x_orig, x_adv):
    """Plots the original RGB, adversarial RGB, and the absolute difference."""
    x_o = x_orig[0].detach().cpu().numpy()
    x_a = x_adv[0].detach().cpu().numpy()

    # RGB channels in Sentinel-2 tensor (B04, B03, B02 -> index 3, 2, 1)
    rgb_orig = np.stack([x_o[3], x_o[2], x_o[1]], axis=-1)
    rgb_adv  = np.stack([x_a[3], x_a[2], x_a[1]], axis=-1)

    scale = np.percentile(rgb_orig, 99)
    rgb_orig = np.clip(rgb_orig / scale, 0, 1)
    rgb_adv  = np.clip(rgb_adv / scale, 0, 1)

    diff = np.abs(rgb_adv - rgb_orig)
    vmax = np.percentile(diff, 99)

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_orig)
    plt.title("Original RGB")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(rgb_adv)
    plt.title("Adversarial RGB")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(diff, cmap='hot', vmin=0, vmax=vmax)
    plt.title("Adversarial Noise (Difference)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def plot_spectral_indices_grid(x_orig):
    """
    Plots a 1x4 grid for a single patch: [RGB, NDVI, NDWI, NDBI].
    """
    tensor = x_orig.detach().cpu().squeeze().numpy()
    rgb, ndvi, ndwi, ndbi = calculate_indices(tensor)
    
    plots = [rgb, ndvi, ndwi, ndbi]
    titles = ["RGB", "NDVI", "NDWI", "NDBI"]
    colormaps = [None, 'RdYlGn', 'Blues', 'YlOrRd']

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for col in range(4):
        ax = axes[col]
        if col == 0:
            ax.imshow(plots[col], aspect='equal')
        else:
            im = ax.imshow(plots[col], cmap=colormaps[col], vmin=-1, vmax=1, aspect='equal')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(titles[col], fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_edgecolor('black' if col == 0 else 'gray')

    plt.tight_layout()
    plt.show()

# ==============================================================================
# MULTI-CHANNEL VISUALIZATIONS
# ==============================================================================

def plot_all_channels_horizontal(x_orig, x_adv, bands_order, title_prefix="", epsilon=0.01):
    """
    Plots all 12 channels horizontally in a 3x12 grid (Orig, Adv, Noise).
    """
    orig = x_orig.detach().cpu().squeeze().numpy()
    adv = x_adv.detach().cpu().squeeze().numpy()
    diff_raw = adv - orig
    num_channels = orig.shape[0]

    fig, axes = plt.subplots(3, num_channels, figsize=(26, 8))
    fig.suptitle(f"{title_prefix} - 12-Channel Adversarial Tensor", fontsize=18, weight='bold', y=1.05)

    for i in range(num_channels):
        band_name = bands_order[i]

        axes[0, i].imshow(orig[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f"Orig\n{band_name}", fontsize=14)
        axes[0, i].axis('off')

        axes[1, i].imshow(adv[i], cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f"Adv\n{band_name}", fontsize=14)
        axes[1, i].axis('off')

        img_plot = axes[2, i].imshow(diff_raw[i], cmap='bwr', vmin=-epsilon, vmax=epsilon)
        axes[2, i].set_title(f"Noise\n{band_name}", fontsize=14)
        axes[2, i].axis('off')

        cbar = fig.colorbar(img_plot, ax=axes[2, i], orientation='horizontal', fraction=0.05, pad=0.05)
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.xaxis.set_tick_params(rotation=45)

    plt.tight_layout()
    plt.show()

# ==============================================================================
# SEMANTIC SHIFT (ADVERSARIAL INDICES)
# ==============================================================================

def plot_semantic_shift_grid(t_orig_list, t_adv_list, labels):
    """
    Plots a 3x3 grid showing the Semantic Shift for indices.
    Expects lists of tensors for [Forest, Water, City] to target [NDVI, NDWI, NDBI].
    """
    def calc_ndvi(t): return (t[7] - t[3]) / (t[7] + t[3] + 1e-8)
    def calc_ndwi(t): return (t[2] - t[7]) / (t[2] + t[7] + 1e-8)
    def calc_ndbi(t): return (t[10] - t[7]) / (t[10] + t[7] + 1e-8)

    funcs = [calc_ndvi, calc_ndwi, calc_ndbi]
    cmaps = ["RdYlGn", "Blues", "YlOrRd"]
    idx_names = ["NDVI", "NDWI", "NDBI"]

    fig, axes = plt.subplots(3, 3, figsize=(10, 10), gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

    for row_idx in range(3):
        t_orig = t_orig_list[row_idx].detach().cpu().squeeze().numpy()
        t_adv = t_adv_list[row_idx].detach().cpu().squeeze().numpy()
        
        idx_orig = funcs[row_idx](t_orig)
        idx_adv = funcs[row_idx](t_adv)
        idx_delta = idx_adv - idx_orig
        vmax_noise = np.max(np.abs(idx_delta))

        # 1. Clean Index
        axes[row_idx, 0].imshow(idx_orig, cmap=cmaps[row_idx], vmin=-1.0, vmax=1.0)
        axes[row_idx, 0].axis('off')
        if row_idx == 0: axes[row_idx, 0].set_title("Clean Index", fontsize=12)

        # 2. Adversarial Index
        axes[row_idx, 1].imshow(idx_adv, cmap=cmaps[row_idx], vmin=-1.0, vmax=1.0)
        axes[row_idx, 1].axis('off')
        if row_idx == 0: axes[row_idx, 1].set_title("Adversarial Index", fontsize=12)

        # 3. Semantic Shift
        axes[row_idx, 2].imshow(idx_delta, cmap="bwr", vmin=-vmax_noise, vmax=vmax_noise)
        axes[row_idx, 2].axis('off')
        if row_idx == 0: axes[row_idx, 2].set_title("Semantic Shift (Delta)", fontsize=12)

        # Labeling the rows
        axes[row_idx, 0].text(-0.1, 0.5, f"{labels[row_idx]}\n({idx_names[row_idx]})", 
                              transform=axes[row_idx, 0].transAxes, 
                              fontsize=12, va='center', ha='right', rotation=90)

    plt.show()
