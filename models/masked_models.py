from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLayer(nn.Module):
    """Wraps a layer with a learnable binary mask."""
    
    def __init__(self, layer, init_mask_value=1.0):
        super(MaskedLayer, self).__init__()
        self.layer = layer
        
        # Freeze the original layer weights
        for param in self.layer.parameters():
            param.requires_grad = False
        
        # Create learnable mask in logit space
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            mask_shape = layer.weight.shape
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")
        
        # Initialize with logits for better gradient flow
        init_logit = torch.log(torch.tensor(init_mask_value / (1 - init_mask_value + 1e-8)))
        # self.mask_logits = nn.Parameter(torch.full(mask_shape, init_logit))
        self.mask_logits = nn.Parameter(torch.ones(mask_shape))
    
    def forward(self, x):
        mask = self.mask_logits
        
        # Apply mask to weights
        if isinstance(self.layer, nn.Conv2d):
            masked_weight = self.layer.weight * mask
            return F.conv2d(
                x, 
                masked_weight, 
                self.layer.bias,
                self.layer.stride,
                self.layer.padding,
                self.layer.dilation,
                self.layer.groups
            )
        elif isinstance(self.layer, nn.Linear):
            # print the shapes
            masked_weight = self.layer.weight * mask
            return F.linear(x, masked_weight, self.layer.bias)
    
    def clamp_mask(self):
        self.mask_logits.data = torch.clamp(self.mask_logits.data, min=0, max=1)
    
    def get_mask(self):
        """Get the actual mask values."""
        return self.mask_logits
    
    def get_binary_mask(self, threshold=0.5):
        """Get binarized version of the mask."""
        return (self.get_mask() > threshold).float()

class BinaryMaskedModel(nn.Module):
    """Wraps a model with learnable binary masks on all conv/linear layers."""
    
    def __init__(self, base_model, init_mask_value=1.0):
        super(BinaryMaskedModel, self).__init__()
        
        # Clone the model structure
        self.base_model = base_model
        
        # Freeze all base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Replace layers with masked versions
        self._replace_layers_with_masked(self.base_model, init_mask_value)
        
        # Collect all masked layers for easy access
        self.masked_layers = []
        self._collect_masked_layers(self.base_model)
    
    def _replace_layers_with_masked(self, module, init_mask_value):
        """Recursively replace Conv2d and Linear layers with MaskedLayer."""
        for name, child in module.named_children():
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                setattr(module, name, MaskedLayer(child, init_mask_value))
            else:
                self._replace_layers_with_masked(child, init_mask_value)
    
    def _collect_masked_layers(self, module):
        """Collect all MaskedLayer instances."""
        for child in module.children():
            if isinstance(child, MaskedLayer):
                self.masked_layers.append(child)
            else:
                self._collect_masked_layers(child)
    
    def forward(self, x):
        return self.base_model(x)
    
    def get_mask_loss(self, p=0.5):
        """L_p norm of masks for sparsity regularization."""
        total_loss = 0
        for layer in self.masked_layers:
            mask = layer.get_mask()
            # L_p norm (p=0.5 encourages sparsity more than L1)
            total_loss += torch.sum(torch.pow(mask, p))
        return total_loss
    
    def get_binary_masks(self, threshold=0.5):
        """Get binary masks for all layers."""
        binary_masks = []
        for layer in self.masked_layers:
            binary_masks.append(layer.get_binary_mask(threshold))
        return binary_masks
    
    def clamp_all_masks(self):
        for layer in self.masked_layers:
            layer.clamp_mask()
    
    def get_sparsity(self, threshold=0.5):
        """Calculate current sparsity level of the model."""
        total_params = 0
        total_zeros = 0
        
        for layer in self.masked_layers:
            mask = layer.get_mask()
            total_params += mask.numel()
            total_zeros += (mask < threshold).sum().item()
        
        return total_zeros / total_params
    
    
    def freeze_masks(self, threshold=0.5):
        """Convert masks to binary and freeze them (for deployment)."""
        with torch.no_grad():
            for layer in self.masked_layers:
                binary_mask = layer.get_binary_mask(threshold)
                # Replace the learnable mask with fixed binary values
                layer.mask_logits.data = torch.log(binary_mask + 1e-8) - torch.log(1 - binary_mask + 1e-8)
                layer.mask_logits.requires_grad = False

