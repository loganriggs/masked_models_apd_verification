import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from models.masked_models import BinaryMaskedModel
from datasets.data_loaders import get_binary_classification_loader

# Example usage for training:
def train_with_masks(masked_model, train_loader, val_loader, config):
    # mask_params = [p for p in masked_model.parameters() if p.requires_grad]

    masked_model = masked_model.to(config['device'])
    mask_params = [param for name, param in masked_model.named_parameters() if 'mask_logits' in name]

    print("------------------------------------------------")
    print("noise scale is", config['noise_scale'])
    print("------------------------------------------------")
    # print(mask_params[0])
    # print("mask params shape")
    # print(len(mask_params))

    optimizer = torch.optim.Adam(mask_params, lr=config['mask_lr'])
    criterion = nn.CrossEntropyLoss()
    for epoch in range(config['max_epochs']):
        best_val_acc = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["max_epochs"]}')
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        idx = 0
        print(config)
        masked_model.train()
        for inputs, targets in pbar:
            inputs, targets = inputs.to(config['device']), targets.to(config['device'])
            noise_scale = config['noise_scale']
            if noise_scale > 0:
                inputs = inputs + torch.randn_like(inputs) * noise_scale

            optimizer.zero_grad()
            
            # Forward pass
            outputs = masked_model(inputs)
            
            # Calculate losses
            task_loss = criterion(outputs, targets)
            mask_loss = config['sparsity_weight'] * masked_model.get_mask_loss(p=config['p'])
            # mask_loss = masked_model.
            total_loss = task_loss +  mask_loss
            
            # # Backward pass
            total_loss.backward()

            # Check gradients
            # for name, param in masked_model.named_parameters():
            #     if 'mask_logits' in name and param.grad is not None:
            #         print(f"{name}: grad_mean={param.grad.mean().item():.6f}, grad_std={param.grad.std().item():.6f}")

            optimizer.step()
            # ensure the mask is in the range of 0 and 1
            with torch.no_grad():
                masked_model.clamp_all_masks()
            # raise Exception("stop here")

            train_loss += total_loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item() 
            
            # Update progress bar
            current_sparsity = masked_model.get_sparsity()
            pbar.set_postfix({
                'task loss': f'{task_loss.item():.3f}',
                'mask loss': f'{mask_loss.item():.3f}',
                'acc': f'{100.*train_correct/train_total:.1f}%',
                'sparsity': f'{current_sparsity:.1%}'
            })
        
        # Print statistics
        sparsity = masked_model.get_sparsity()
        print(f"Epoch {epoch}: Sparsity = {sparsity:.2%}")
                # Validation
        masked_model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(config['device']), targets.to(config['device'])
                outputs = masked_model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        current_sparsity = masked_model.get_sparsity()
        
        print(f'Epoch {epoch+1}: Val Acc: {val_acc:.1f}%, Sparsity: {current_sparsity:.1%}')
        
        # # Early stopping
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        #     if patience_counter >= config['patience']:
        #         print(f'Early stopping at epoch {epoch+1}')
        #         break
        
        # # Check if we've reached target sparsity with good accuracy
        # if current_sparsity >= config['sparsity_target'] and val_acc >= config['min_accuracy']:
        #     print(f'Reached target sparsity {current_sparsity:.1%} with accuracy {val_acc:.1f}%')
        #     break
    
    return masked_model

def pairwise_pruning(model, target_class, comparison_classes, dataset_name, config):
    """
    For a target class, create multiple binary classifiers.
    Returns masks for each pairwise task.
    """
    masks_list = []
    
    for comp_class in comparison_classes:
        print(f"\nTraining binary classifier: class {target_class} vs class {comp_class}")
        
        # Create binary classification data loaders
        train_loader = get_binary_classification_loader(
            dataset_name, target_class, comp_class, train=True, 
            batch_size=config.get('batch_size', 128)
        )
        val_loader = get_binary_classification_loader(
            dataset_name, target_class, comp_class, train=False, 
            batch_size=config.get('batch_size', 128)
        )
        
        
        # Modify the model's output layer for binary classification
        # print(masked_model.base_model)
        if hasattr(model, 'linear'):
            original_linear = model.linear
            in_features = original_linear.in_features
            
            # Create new linear layer with only 2 outputs
            new_linear = nn.Linear(in_features, 2)
            
            # Copy weights and biases for target_class -> index 0, comp_class -> index 1
            new_linear.weight.data[0] = original_linear.weight.data[target_class]
            new_linear.weight.data[1] = original_linear.weight.data[comp_class]
            new_linear.bias.data[0] = original_linear.bias.data[target_class]
            new_linear.bias.data[1] = original_linear.bias.data[comp_class]
            
            # Replace the layer inside the MaskedLayer
            model.linear = new_linear

        elif hasattr(model, 'fc'):
            original_fc = model.fc
            in_features = original_fc.in_features
            
            # Create new fc layer with only 2 outputs
            new_fc = nn.Linear(in_features, 2)
            
            # Copy weights and biases for target_class -> index 0, comp_class -> index 1
            new_fc.weight.data[0] = original_fc.weight.data[target_class]
            new_fc.weight.data[1] = original_fc.weight.data[comp_class]
            new_fc.bias.data[0] = original_fc.bias.data[target_class]
            new_fc.bias.data[1] = original_fc.bias.data[comp_class]
            
            # Replace the fc layer
            model.fc = new_fc
        # Create masked model (reinitialize masks for each task)
        masked_model = BinaryMaskedModel(model, init_mask_value=1.0)

        # Train the masks
        trained_model = train_with_masks(masked_model.to(config['device']), train_loader, val_loader, config)
        
        # Extract the learned masks
        binary_masks = trained_model.get_binary_masks(threshold=0.5)
        masks_list.append(binary_masks)
    
    return masks_list

def run_pairwise_pruning(model, dataset, target_class, num_comparisons=8, config=None):
    """Main function to run pairwise pruning experiments."""
    if config is None:
        config = {
            'mask_lr': 0.001,
            'max_epochs': 50,
            'sparsity_weight': 0.001,
            'sparsity_target': 0.9,
            'min_accuracy': 80.0,
            'patience': 5,
            'batch_size': 128
        }
    
    # Get dataset info
    from datasets.data_loaders import get_dataset_info
    dataset_info = get_dataset_info(dataset)
    num_classes = dataset_info['num_classes']
    
    # Select comparison classes (all except target)
    all_classes = list(range(num_classes))
    all_classes.remove(target_class)
    comparison_classes = np.random.choice(all_classes, 
                                        min(num_comparisons, len(all_classes)), 
                                        replace=False)
    
    print(f"Target class: {target_class} ({dataset_info['classes'][target_class]})")
    print(f"Comparison classes: {comparison_classes.tolist()}")
    
    # Run pairwise pruning
    masks = pairwise_pruning(model, target_class, comparison_classes, dataset, config)
    
    return masks

