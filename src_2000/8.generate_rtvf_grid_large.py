"""
Generate 4x4 Grid Plots for R-TVF Results (SRC_2000 LARGE Version)
==================================================================
Generates a 4x4 grid of cropped R-TVF plots for 16 representative stars
of each type (HADS/LADS) for the paper.
LARGE VERSION: Uses _large.png inputs and outputs _large.png grids.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Adjusted to SRC_2000 paths
HADS_PLOT_DIR = os.path.join(BASE_DIR, 'src_2000/results_2000/2_cross_validation/plots_hads')
LADS_PLOT_DIR = os.path.join(BASE_DIR, 'src_2000/results_2000/2_cross_validation/plots_lads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'src_2000/results_2000/4_analysis/plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fixed settings
CROP_TOP_PERCENT = 0.11
DPI = 600

# 16 Selected Stars for HADS (Filtered list)
HADS_STARS = [
    'AD_Cmilc', 'AE_Umalc', 'BS_Aqrlc', 'DE_Laclc',
    'V1393_Cenlc', 'GW_Umalc', 'HD_146953lc', 'Rho_Puplc',
    'RS_Grulc', 'RY_Leplc', 'SS_Psclc', 'SX_Phelc',
    'V1162_Orilc', 'V1384_Tau_lc', 'DX_Cetlc', 'V1719_Cyglc'
]

# 16 Selected Stars for LADS
LADS_STARS = [
    'TIC_100531058lc', 'TIC_102192161lc', 'TIC_452982723lc', 'TIC_11361473lc',
    'TIC_121729614lc', 'TIC_122615966lc', 'TIC_12524129_lc', 'TIC_137341551lc',
    'TIC_137796620lc', 'TIC_159647185lc', 'TIC_161172103lc', 'TIC_172193026lc',
    'TIC_253296458lc', 'TIC_196921106lc', 'TIC_242944780lc', 'TIC_246902545lc'
]

def format_star_name(fname):
    """Convert filename to readable star name."""
    name = fname.replace('lc', '').replace('_', ' ')
    if name.startswith('TIC'):
        pass
    return name.strip()

def generate_4x4_grid(star_list, plot_dir, output_filename, title_main):
    """Generate 4x4 grid plot."""
    print(f"Generating {output_filename}...")
    
    # Wider figure for elongated plots (24 width, 12 height)
    fig, axes = plt.subplots(4, 4, figsize=(24, 12))
    fig.suptitle(title_main, fontsize=24, fontweight='bold', y=0.98)
    
    axes_flat = axes.flatten()
    
    for i, star_base in enumerate(star_list):
        ax = axes_flat[i]
        
        # Large Image Path
        img_name = f"{star_base}_R-TVF_fold0_large.png"
        img_path = os.path.join(plot_dir, img_name)
        
        display_name = format_star_name(star_base)
        
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            
            # Crop top portion
            height = img.shape[0]
            crop_pixels = int(height * CROP_TOP_PERCENT)
            img_cropped = img[crop_pixels:, :, :]
            
            ax.imshow(img_cropped, aspect='auto')
            ax.axis('off')
            ax.set_title(display_name, fontsize=12, fontweight='bold', pad=5)
        else:
            ax.text(0.5, 0.5, f'Missing:\n{display_name}', 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            print(f"  Warning: Missing {img_path}")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.1, wspace=0.1)
    
    save_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    print("="*60)
    print("Generating 4x4 Grid Plots for R-TVF (SRC_2000 LARGE Version)")
    print("="*60)
    
    generate_4x4_grid(HADS_STARS, HADS_PLOT_DIR, "rtvf_grid_hads_large.png", "R-TVF Results: HADS Stars")
    generate_4x4_grid(LADS_STARS, LADS_PLOT_DIR, "rtvf_grid_lads_large.png", "R-TVF Results: LADS Stars")
    
    print("\nDone.")
