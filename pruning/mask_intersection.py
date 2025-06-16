import torch
import numpy as np

def intersect_masks(mask_list, method='multiply', threshold=0.5):
    """
    Combine masks from multiple pairwise tasks.
    
    Args:
        mask_list: List of mask lists (each from a pairwise task)
        method: 'multiply' (AND), 'average', or 'minimum'
        threshold: Threshold for binarization (if using average)
    
    Returns:
        Combined masks for each layer
    """
    if not mask_list:
        return []
    
    num_layers = len(mask_list[0])
    combined_masks = []
    
    for layer_idx in range(num_layers):
        # Collect masks for this layer from all pairwise tasks
        layer_masks = [masks[layer_idx] for masks in mask_list]
        
        if method == 'multiply':
            # AND operation: only keep weights that are active in ALL tasks
            combined = torch.stack(layer_masks).prod(dim=0)
        elif method == 'average':
            # Average and threshold
            combined = torch.stack(layer_masks).mean(dim=0)
            combined = (combined >= threshold).float()
        elif method == 'minimum':
            # Take minimum value across masks
            combined = torch.stack(layer_masks).min(dim=0)[0]
        else:
            raise ValueError(f"Unknown intersection method: {method}")
        
        combined_masks.append(combined)
    
    return combined_masks

def combine_masks(mask_list, method='multiply'):
    """
    Alternative interface for mask combination.
    """
    return intersect_masks(mask_list, method=method)

def analyze_mask_agreement(mask_list):
    """
    Analyze how much masks agree across different pairwise tasks.
    
    Returns statistics about mask agreement.
    """
    if not mask_list:
        return {}
    
    num_layers = len(mask_list[0])
    agreement_stats = []
    
    for layer_idx in range(num_layers):
        layer_masks = torch.stack([masks[layer_idx] for masks in mask_list])
        
        # Calculate agreement: how many tasks agree on each weight
        agreement = layer_masks.mean(dim=0)
        
        stats = {
            'layer_idx': layer_idx,
            'full_agreement': (agreement == 1.0).sum().item(),  # All tasks keep this weight
            'no_agreement': (agreement == 0.0).sum().item(),    # All tasks prune this weight
            'partial_agreement': ((agreement > 0) & (agreement < 1)).sum().item(),
            'mean_agreement': agreement.mean().item(),
            'total_params': agreement.numel()
        }
        agreement_stats.append(stats)
    
    return agreement_stats
