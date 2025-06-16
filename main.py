
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm

from models import ResNet20, BinaryMaskedModel
from datasets import get_cifar10_loaders, get_fashion_mnist_loaders, get_svhn_loaders
from pruning.pairwise_pruning import run_pairwise_pruning
from pruning.mask_intersection import intersect_masks
from evaluation import evaluate_accuracy_sparsity, plot_accuracy_sparsity_curve
from utils import plot_training_history

def train_model(args):
    """Train a ResNet-20 model from scratch."""
    print(f"Training ResNet-20 on {args.dataset}")
    
    # Get data loaders
    if args.dataset == 'cifar10':
        train_loader, test_loader, class_names = get_cifar10_loaders(args.batch_size)
        num_classes = 10
        in_channels = 3
    elif args.dataset == 'fashion_mnist':
        train_loader, test_loader, class_names = get_fashion_mnist_loaders(args.batch_size)
        num_classes = 10
        in_channels = 1
    elif args.dataset == 'svhn':
        train_loader, test_loader, class_names = get_svhn_loaders(args.batch_size)
        num_classes = 10
        in_channels = 3
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Create model
    model = ResNet20(num_classes=num_classes)
    
    # Modify first conv layer for Fashion-MNIST
    if args.dataset == 'fashion_mnist':
        model.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training history
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    
    # Training loop
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.3f}', 
                             'acc': f'{100.*correct/total:.1f}%'})
        
        # Validate
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        test_acc = 100. * test_correct / test_total
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(test_acc)
        
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.1f}%, Test Acc: {test_acc:.1f}%')
        
        scheduler.step()
    
    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    save_path = f'checkpoints/resnet20_{args.dataset}_acc{test_acc:.1f}.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    # Plot training history
    plot_training_history(history, save_path=f'plots/training_history_{args.dataset}.png')
    
    return model

def run_pruning(args):
    """Run pairwise pruning experiments."""
    print(f"Running pairwise pruning on {args.dataset}")
    
    # Load model
    if args.dataset == 'cifar10':
        _, test_loader, class_names = get_cifar10_loaders(args.batch_size)
        num_classes = 10
    elif args.dataset == 'fashion_mnist':
        _, test_loader, class_names = get_fashion_mnist_loaders(args.batch_size)
        num_classes = 10
    elif args.dataset == 'svhn':
        _, test_loader, class_names = get_svhn_loaders(args.batch_size)
        num_classes = 10
    
    model = ResNet20(num_classes=num_classes)
    
    # Modify for Fashion-MNIST
    if args.dataset == 'fashion_mnist':
        model.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Load checkpoint
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
        print(f"Loaded model from {args.checkpoint}")
    else:
        print("Warning: No checkpoint provided, using random initialization")
    
    print(args)
    print("about to run pruning")
    print(args.p)
    # Pruning configuration
    config = {
        'mask_lr': args.mask_lr,
        'max_epochs': args.prune_epochs,
        'sparsity_weight': args.sparsity_weight,
        'sparsity_target': args.sparsity_target,
        'min_accuracy': args.min_accuracy,
        'patience': 5,
        'batch_size': args.batch_size,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'p': args.p,
        'noise_scale': args.noise_scale,
    }
    
    # Run pairwise pruning
    masks = run_pairwise_pruning(
        model=model,
        dataset=args.dataset,
        target_class=args.target_class,
        num_comparisons=args.num_comparisons,
        config=config
    )
    
    # Combine masks
    final_masks = intersect_masks(masks, method=args.intersection_method)
    
    # Evaluate accuracy at different sparsity levels
    results = evaluate_accuracy_sparsity(model, final_masks, test_loader)
    
    # Plot results
    os.makedirs('plots', exist_ok=True)
    plot_accuracy_sparsity_curve(
        results, 
        save_path=f'plots/accuracy_sparsity_{args.dataset}_class{args.target_class}.png',
        title=f'Accuracy vs Sparsity for {class_names[args.target_class]}'
    )
    
    # Save masks
    os.makedirs('masks', exist_ok=True)
    mask_path = f'masks/{args.dataset}_class{args.target_class}_masks.pth'
    torch.save(final_masks, mask_path)
    print(f"Masks saved to {mask_path}")

def main():
    parser = argparse.ArgumentParser(description='APD Verification System')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model from scratch')
    train_parser.add_argument('--dataset', type=str, default='cifar10',
                            choices=['cifar10', 'fashion_mnist', 'svhn'])
    train_parser.add_argument('--epochs', type=int, default=200)
    train_parser.add_argument('--batch_size', type=int, default=128)
    train_parser.add_argument('--lr', type=float, default=0.1)
    
    # Prune command
    prune_parser = subparsers.add_parser('prune', help='Run pairwise pruning')
    prune_parser.add_argument('--dataset', type=str, default='cifar10',
                            choices=['cifar10', 'fashion_mnist', 'svhn'])
    prune_parser.add_argument('--checkpoint', type=str, required=True,
                            help='Path to model checkpoint')
    prune_parser.add_argument('--target_class', type=int, default=3,
                            help='Target class index')
    prune_parser.add_argument('--num_comparisons', type=int, default=8,
                            help='Number of pairwise comparisons')
    prune_parser.add_argument('--batch_size', type=int, default=128)
    prune_parser.add_argument('--mask_lr', type=float, default=0.01)
    prune_parser.add_argument('--prune_epochs', type=int, default=50)
    prune_parser.add_argument('--sparsity_weight', type=float, default=0.001)
    prune_parser.add_argument('--sparsity_target', type=float, default=0.9)
    prune_parser.add_argument('--min_accuracy', type=float, default=80.0)
    prune_parser.add_argument('--intersection_method', type=str, default='multiply',
                            choices=['multiply', 'average', 'minimum'])
    prune_parser.add_argument('--noise_scale', type=float, default=0.0)
    prune_parser.add_argument('--p', type=float, default=0.5)
    
    args = parser.parse_args()
    print(args)
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'prune':
        run_pruning(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()