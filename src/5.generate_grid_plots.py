"""
Grid Plot Generator - With subtitles and cropped images
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
HADS_PLOT_DIR = os.path.join(BASE_DIR, 'results/2_cross_validation/plots_hads')
LADS_PLOT_DIR = os.path.join(BASE_DIR, 'results/2_cross_validation/plots_lads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results/4_analysis/plots')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Method order and display names
METHODS = ['ARMAX', 'GP-Celerite', 'DHR', 'TBATS', 'FS-BSM', 'R-TVF']
METHOD_DISPLAY_NAMES = {
    'ARMAX': 'ARMAX', 
    'GP-Celerite': 'GP-Celerite',
    'DHR': 'DHR',
    'FS-BSM': 'FS-BSM',
    'TBATS': 'TBATS',
    'R-TVF': 'R-TVF (Ours)'
}

# Fixed fold
FIXED_FOLD = 0

# Percentage of image to crop from top (to remove built-in title)
CROP_TOP_PERCENT = 0.11  # Crop top 11%


def generate_grid_plot(star_name, star_file, plot_dir, output_filename):
    """Generate a 3x2 grid plot with subtitles and cropped images."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))  # Proportional to 4x4 (24,12) -> 2 cols = 12 width, 3 rows = 9 height
    fig.suptitle(
        f"Comparison of Imputation Methods for {star_name}",
        fontsize=20, y=0.98, fontweight='bold'
    )
    
    axes_flat = axes.flatten()
    plot_prefix = star_file.replace('.dat', '').replace(' ', '_')
    
    for i, method in enumerate(METHODS):
        ax = axes_flat[i]
        display_name = METHOD_DISPLAY_NAMES[method]
        
        img_name = f"{plot_prefix}_{method}_fold{FIXED_FOLD}.png"
        img_path = os.path.join(plot_dir, img_name)
        
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            
            # Crop top portion to remove built-in title
            height = img.shape[0]
            crop_pixels = int(height * CROP_TOP_PERCENT)
            img_cropped = img[crop_pixels:, :, :]
            
            ax.imshow(img_cropped, aspect='auto') # Force auto aspect to fill
            ax.axis('off')
            ax.set_title(display_name, fontsize=16, fontweight='bold', pad=4)
        else:
            ax.text(0.5, 0.5, 'Convergence Failed\n(Gap > 1000)', 
                   ha='center', va='center', fontsize=16, color='red', fontweight='bold')
            ax.axis('off')
            ax.set_title(display_name, fontsize=16, fontweight='bold')
            print(f"Warning: Image not found: {img_path}")

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.15, wspace=0.02) # Adjusted for tighter layout
    
    save_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Saved grid plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    print("="*70)
    print("Grid Plot Generator (with subtitles, cropped images)")
    print(f"Cropping top {CROP_TOP_PERCENT*100:.0f}% of each image")
    print("="*70)
    
    display_suffix = "_large"
    
    # HADS
    print("\nProcessing HADS Star: HD 146953...")
    generate_grid_plot(
        "HD 146953 (HADS)", 
        "HD 146953lc.dat", 
        HADS_PLOT_DIR, 
        "grid_comparison_HD146953.png"
    )
    
    # LADS
    print("\nProcessing LADS Star: TIC 452982723...")
    generate_grid_plot(
        "TIC 452982723 (LADS)", 
        "TIC 452982723lc.dat", 
        LADS_PLOT_DIR, 
        "grid_comparison_TIC452982723.png"
    )
    
    print("\nDone!")
