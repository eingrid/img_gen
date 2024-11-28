import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def load_cifar10_data(batch_size=64, val_split=0.1):
    """
    Loads CIFAR-10 dataset with training, validation, and test splits.
    
    Args:
        batch_size (int): Number of samples per batch.
        val_split (float): Proportion of the training set to use for validation.
    
    Returns:
        train_loader, val_loader, test_loader (DataLoader): Data loaders for train, val, and test datasets.
    """
    # Define transformations for training and test data
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the CIFAR-10 training and test datasets
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    # Split training data into train and validation sets
    train_size = int((1 - val_split) * len(train_data))
    val_size = len(train_data) - train_size
    train_data, val_data = random_split(train_data, [train_size, val_size])

    # Wrapping datasets to return image as both input and target
    train_data = ImageToImageDataset(train_data)
    val_data = ImageToImageDataset(val_data)
    test_data = ImageToImageDataset(test_data)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader

class ImageToImageDataset(torch.utils.data.Dataset):
    """
    Wrapper for a dataset to return the image as both input and target, 
    useful for Autoencoder and VAE tasks.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return img, img  # Return the same image as input and target

if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_cifar10_data(batch_size=64)
    print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}, Test batches: {len(test_loader)}")
