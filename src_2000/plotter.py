import matplotlib.pyplot as plt
import numpy as np

def plot_results(t, y_true, y_missing, results, start, end, title="Imputation Results", output_path=None):
    """
    Plots the results of the imputation.
    results: list of tuples (label, y_imp, color)
    """
    pad = 200
    left = max(0, start - pad)
    right = min(len(t), end + pad)
    
    tx = t[left:right]
    y_true_win = y_true[left:right]
    y_miss_win = y_missing[left:right]
    
    # Determine grid size based on number of results
    n_methods = len(results)
    cols = 2
    rows = (n_methods + cols - 1) // cols
    if rows < 1: rows = 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows), sharex=True)
    
    # Ensure axes is iterable
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
        
    for i, ax in enumerate(axes):
        if i < n_methods:
            label, y_imp, color = results[i]
            
            # Plot original
            ax.plot(tx, y_true_win, lw=1.0, color="black", ls="--", label="Original")
            
            # Plot missing (with gap)
            # We plot the missing data. Since it has NaNs in the gap, it will show a break.
            ax.plot(tx, y_miss_win, lw=1.0, color="red", ls="--", label="Input (NaN)")
            
            # Plot imputed
            y_imp = np.asarray(y_imp, dtype=float)
            ax.plot(tx, y_imp[left:right], lw=1.6, color=color, label=label)
            
            # Highlight gap
            ax.axvspan(t[start], t[end-1], color="lightgray", alpha=0.5, label=f"Gap N={end-start}")
            
            ax.grid(True)
            ax.set_ylabel("Flux")
            ax.legend(loc="upper left", fontsize=9)
            ax.set_title(label)
        else:
            ax.axis('off')
            
    # Set xlabel for bottom axes
    # This logic is a bit simple, might need adjustment if some axes are off
    for ax in axes:
        if ax.get_subplotspec().is_last_row():
             ax.set_xlabel("Time (HJD)")
        
    fig.suptitle(title, y=0.995, fontsize=15)
    fig.tight_layout(rect=[0,0,1,0.98])
    
    if output_path:
        fig.savefig(output_path, dpi=300)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close(fig)
