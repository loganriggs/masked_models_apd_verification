import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def evaluate_model_with_masks(model, masks, test_loader, device='cuda'):
    """Evaluate model accuracy with given masks applied."""
    model.eval()
    model = model.to(device)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        # Apply masks to model weights
        mask_idx = 0
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                if mask_idx < len(masks):
                    module.weight.data *= masks[mask_idx].to(device)
                    mask_idx += 1
        
        # Evaluate
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

def evaluate_accuracy_sparsity(model, masks, test_loader, sparsity_levels=None):
    """
    Evaluate accuracy at different sparsity levels.
    
    Args:
        model: The base model
        masks: Binary masks from pruning
        test_loader: Test data loader
        sparsity_levels: List of sparsity levels to evaluate
    
    Returns:
        Dictionary with sparsity levels and corresponding accuracies
    """
    if sparsity_levels is None:
        sparsity_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {'sparsity': [], 'accuracy': []}
    
    # Get baseline accuracy (no pruning)
    baseline_acc = evaluate_model_with_masks(model, 
                                            [torch.ones_like(m) for m in masks], 
                                            test_loader, device)
    print(f"Baseline accuracy: {baseline_acc:.2f}%")
    
    # Evaluate at different sparsity levels
    for sparsity in tqdm(sparsity_levels, desc="Evaluating sparsity levels"):
        # Create masks at this sparsity level
        pruned_masks = []
        for mask in masks:
            # Get magnitude of weights
            mask_flat = mask.flatten()
            num_params = mask_flat.numel()
            num_zeros = int(num_params * sparsity)
            
            # Find threshold for desired sparsity
            if num_zeros > 0:
                threshold = torch.kthvalue(mask_flat, num_zeros)[0]
                pruned_mask = (mask > threshold).float()
            else:
                pruned_mask = torch.ones_like(mask)
            
            pruned_masks.append(pruned_mask)
        
        # Evaluate accuracy
        acc = evaluate_model_with_masks(model.clone(), pruned_masks, test_loader, device)
        
        results['sparsity'].append(sparsity)
        results['accuracy'].append(acc)
        
        print(f"Sparsity: {sparsity:.1%}, Accuracy: {acc:.2f}%")
    
    return results

def plot_accuracy_sparsity_curve(results, save_path=None, title="Accuracy vs Sparsity"):
    """Plot accuracy vs sparsity curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(results['sparsity'], results['accuracy'], 'o-', linewidth=2, markersize=8)
    plt.xlabel('Sparsity Level', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add annotations for key points
    for i, (s, a) in enumerate(zip(results['sparsity'], results['accuracy'])):
        if i % 2 == 0:  # Annotate every other point to avoid clutter
            plt.annotate(f'{a:.1f}%', (s, a), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
    
    plt.xlim(-0.05, 1.05)
    plt.ylim(0, 105)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def compare_accuracy_sparsity_curves(results_dict, save_path=None, title="Accuracy vs Sparsity Comparison"):
    """Compare multiple accuracy vs sparsity curves."""
    plt.figure(figsize=(12, 8))
    
    for label, results in results_dict.items():
        plt.plot(results['sparsity'], results['accuracy'], 'o-', 
                linewidth=2, markersize=8, label=label)
    
    plt.xlabel('Sparsity Level', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.05, 1.05)
    plt.ylim(0, 105)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
