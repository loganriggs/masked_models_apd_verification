
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

FASHION_MNIST_CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

SVHN_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def get_cifar10_loaders(batch_size=128, num_workers=4):
    """Get CIFAR-10 data loaders."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                           shuffle=True, num_workers=num_workers)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers)
    
    return trainloader, testloader, CIFAR10_CLASSES

def get_fashion_mnist_loaders(batch_size=128, num_workers=4):
    """Get Fashion-MNIST data loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                           shuffle=True, num_workers=num_workers)
    
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                               download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers)
    
    return trainloader, testloader, FASHION_MNIST_CLASSES

def get_svhn_loaders(batch_size=128, num_workers=4):
    """Get SVHN data loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.SVHN(root='./data', split='train',
                                        download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                           shuffle=True, num_workers=num_workers)
    
    testset = torchvision.datasets.SVHN(root='./data', split='test',
                                       download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers)
    
    return trainloader, testloader, SVHN_CLASSES

def get_binary_classification_loader(dataset, class1, class2, train=True, batch_size=128):
    """Create a balanced binary classification dataset from two classes."""
    if dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform = transform_train if train else transform_test
        base_dataset = torchvision.datasets.CIFAR10(root='./data', train=train,
                                                   download=True, transform=transform)
    elif dataset == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        base_dataset = torchvision.datasets.FashionMNIST(root='./data', train=train,
                                                        download=True, transform=transform)
    elif dataset == 'svhn':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        split = 'train' if train else 'test'
        base_dataset = torchvision.datasets.SVHN(root='./data', split=split,
                                                download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Get indices for both classes
    targets = np.array(base_dataset.targets if hasattr(base_dataset, 'targets') else base_dataset.labels)
    class1_indices = np.where(targets == class1)[0]
    class2_indices = np.where(targets == class2)[0]
    
    # Balance the dataset
    min_samples = min(len(class1_indices), len(class2_indices))
    class1_indices = np.random.choice(class1_indices, min_samples, replace=False)
    class2_indices = np.random.choice(class2_indices, min_samples, replace=False)
    
    # Combine indices
    all_indices = np.concatenate([class1_indices, class2_indices])
    np.random.shuffle(all_indices)
    
    # Create subset
    subset = Subset(base_dataset, all_indices)
    
    # Create a wrapper to convert labels to binary
    class BinaryDataset(torch.utils.data.Dataset):
        def __init__(self, subset, class1, class2):
            self.subset = subset
            self.class1 = class1
            self.class2 = class2
        
        def __len__(self):
            return len(self.subset)
        
        def __getitem__(self, idx):
            data, target = self.subset[idx]
            # Convert to binary: class1 -> 0, class2 -> 1
            binary_target = 0 if target == self.class1 else 1
            return data, binary_target
    
    binary_dataset = BinaryDataset(subset, class1, class2)
    return DataLoader(binary_dataset, batch_size=batch_size, shuffle=True)

def get_dataset_info(dataset_name):
    """Get information about a dataset."""
    if dataset_name == 'cifar10':
        return {
            'num_classes': 10,
            'classes': CIFAR10_CLASSES,
            'input_shape': (3, 32, 32),
            'num_train': 50000,
            'num_test': 10000
        }
    elif dataset_name == 'fashion_mnist':
        return {
            'num_classes': 10,
            'classes': FASHION_MNIST_CLASSES,
            'input_shape': (1, 28, 28),
            'num_train': 60000,
            'num_test': 10000
        }
    elif dataset_name == 'svhn':
        return {
            'num_classes': 10,
            'classes': SVHN_CLASSES,
            'input_shape': (3, 32, 32),
            'num_train': 73257,
            'num_test': 26032
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
