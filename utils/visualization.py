
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_training_history(history, save_path=None):
    """Plot training history including loss and accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy')
    if 'val_acc' in history:
        ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_filters(model, layer_name, num_filters=16):
    """Visualize convolutional filters from a specific layer."""
    # Find the target layer
    target_layer = None
    for name, module in model.named_modules():
        if name == layer_name and isinstance(module, torch.nn.Conv2d):
            target_layer = module
            break
    
    if target_layer is None:
        print(f"Layer {layer_name} not found or is not a Conv2d layer")
        return
    
    # Get filters
    filters = target_layer.weight.data.cpu()
    num_filters = min(num_filters, filters.shape[0])
    
    # Create subplot grid
    rows = int(np.sqrt(num_filters))
    cols = int(np.ceil(num_filters / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten() if num_filters > 1 else [axes]
    
    for i in range(num_filters):
        ax = axes[i]
        
        # Get the filter
        filter_tensor = filters[i]
        
        # Handle different numbers of input channels
        if filter_tensor.shape[0] == 1:
            # Grayscale
            img = filter_tensor[0].numpy()
        elif filter_tensor.shape[0] == 3:
            # RGB - transpose to HWC format
            img = filter_tensor.permute(1, 2, 0).numpy()
            # Normalize to [0, 1]
            img = (img - img.min()) / (img.max() - img.min())
        else:
            # Multiple channels - show first channel
            img = filter_tensor[0].numpy()
        
        ax.imshow(img, cmap='viridis' if len(img.shape) == 2 else None)
        ax.set_title(f'Filter {i}')
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(num_filters, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Filters from {layer_name}')
    plt.tight_layout()
    plt.show()

def plot_class_distribution(dataset_loader, dataset_name, class_names):
    """Plot the distribution of classes in a dataset."""
    class_counts = np.zeros(len(class_names))
    
    for _, targets in dataset_loader:
        for target in targets:
            class_counts[target] += 1
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(class_names)), class_counts)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title(f'{dataset_name} Class Distribution')
    
    # Add value labels on bars
    for bar, count in zip(bars, class_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
