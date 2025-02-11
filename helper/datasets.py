from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# TODO Add documentation

# Transformer Definitions
transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to mean 0.5 and std 0.5
])

transform_fmnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to mean 0.5 and std 0.5
])

# Upscaling required for larger scale models
transform_cifar10 = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))  # CIFAR-10 normalization
])


def get_dataset_loader(dataset_name, batch_size=64):
    # Load the datasets
    if dataset_name == "mnist":
        train_dataset = datasets.MNIST(
            root='./data', train=True, transform=transform_mnist, download=True)
        test_dataset = datasets.MNIST(
            root='./data', train=False, transform=transform_mnist, download=True)
    elif dataset_name == "fmnist":
        train_dataset = datasets.FashionMNIST(
            root='./data', train=True, transform=transform_fmnist, download=True)
        test_dataset = datasets.FashionMNIST(
            root='./data', train=False, transform=transform_fmnist, download=True)
    elif dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root='./data', train=True, transform=transform_cifar10, download=True)
        test_dataset = datasets.CIFAR10(
            root='./data', train=False, transform=transform_cifar10, download=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64,
                             shuffle=False, num_workers=4)

    return train_loader, test_loader
