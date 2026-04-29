import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# soft MNIST: pixels are continuous in [0,1]; treated as Bernoulli means in the BCE loss
def get_loaders(batch_size=128, root="./mnist_data", num_workers=0):
    tfm = transforms.ToTensor()
    train = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    test = datasets.MNIST(root=root, train=False, download=True, transform=tfm)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False,
                             drop_last=False, num_workers=num_workers)
    return train_loader, test_loader


def fixed_test_batch(test_loader, n=8):
    x, y = next(iter(test_loader))
    return x[:n], y[:n]
