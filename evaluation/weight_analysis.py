import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_masks(masks, layer_names=None, save_path=None):
    """Visualize binary masks as heatmaps."""
    num_layers = len(masks)
    
    # Create subplots
    fig, axes = plt.subplots(2, (num_layers + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, mask in enumerate(masks):
        ax = axes[i]
        
        # Reshape mask for visualization
        if len(mask.shape) == 4:  # Conv2d
            # Show first output channel
            mask_2d = mask[0, 0].cpu().numpy()
        elif len(mask.shape) == 2:  # Linear
            mask_2d = mask.cpu().numpy()
        else:
            continue
        
        # Plot heatmap
        sns.heatmap(mask_2d, ax=ax, cmap='RdBu_r', cbar=True, 
                   vmin=0, vmax=1, square=True)
        
        # Set title
        if layer_names and i < len(layer_names):
            ax.set_title(f'{layer_names[i]}')
        else:
            ax.set_title(f'Layer {i+1}')
        
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    # Hide extra subplots
    for i in range(num_layers, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_weight_distribution(model, masks):
    """Analyze the distribution of pruned vs retained weights."""
    weight_magnitudes = []
    mask_values = []
    
    mask_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if mask_idx < len(masks):
                weights = module.weight.data.cpu().numpy().flatten()
                mask = masks[mask_idx].cpu().numpy().flatten()
                
                weight_magnitudes.extend(np.abs(weights))
                mask_values.extend(mask)
                mask_idx += 1
    
    weight_magnitudes = np.array(weight_magnitudes)
    mask_values = np.array(mask_values)
    
    # Separate pruned and retained weights
    pruned_weights = weight_magnitudes[mask_values < 0.5]
    retained_weights = weight_magnitudes[mask_values >= 0.5]
    
    # Plot distributions
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(pruned_weights, bins=50, alpha=0.7, label='Pruned', density=True)
    plt.hist(retained_weights, bins=50, alpha=0.7, label='Retained', density=True)
    plt.xlabel('Weight Magnitude')
    plt.ylabel('Density')
    plt.title('Weight Magnitude Distribution')
    plt.legend()
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    # Plot cumulative distributions
    pruned_sorted = np.sort(pruned_weights)
    retained_sorted = np.sort(retained_weights)
    
    plt.plot(pruned_sorted, np.linspace(0, 1, len(pruned_sorted)), 
             label='Pruned', linewidth=2)
    plt.plot(retained_sorted, np.linspace(0, 1, len(retained_sorted)), 
             label='Retained', linewidth=2)
    plt.xlabel('Weight Magnitude')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\nWeight Distribution Statistics:")
    print(f"Pruned weights: mean={np.mean(pruned_weights):.4f}, "
          f"std={np.std(pruned_weights):.4f}, "
          f"median={np.median(pruned_weights):.4f}")
    print(f"Retained weights: mean={np.mean(retained_weights):.4f}, "
          f"std={np.std(retained_weights):.4f}, "
          f"median={np.median(retained_weights):.4f}")
    print(f"Retention rate: {len(retained_weights) / len(weight_magnitudes):.2%}")

def plot_layer_sparsity(masks, layer_names=None):
    """Plot sparsity level for each layer."""
    sparsity_per_layer = []
    
    for mask in masks:
        total_params = mask.numel()
        num_zeros = (mask < 0.5).sum().item()
        sparsity = num_zeros / total_params
        sparsity_per_layer.append(sparsity)
    
    plt.figure(figsize=(10, 6))
    x = range(len(sparsity_per_layer))
    plt.bar(x, sparsity_per_layer)
    
    if layer_names:
        plt.xticks(x, layer_names, rotation=45, ha='right')
    else:
        plt.xticks(x, [f'Layer {i+1}' for i in x])
    
    plt.ylabel('Sparsity Level')
    plt.title('Sparsity by Layer')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for i, v in enumerate(sparsity_per_layer):
        plt.text(i, v + 0.01, f'{v:.2%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

