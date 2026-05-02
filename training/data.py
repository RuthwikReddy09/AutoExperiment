from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms


def build_loaders(config) -> Tuple[DataLoader, DataLoader]:
    data_dir = config["data_dir"]
    batch_size = int(config["batch_size"])
    num_workers = int(config.get("num_workers", 2))

    train_tfms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    eval_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    full_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tfms)
    full_val_source = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=eval_tfms)

    val_size = 5000
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(int(config.get("seed", 42)))
    train_subset, _ = random_split(full_train, [train_size, val_size], generator=generator)
    _, val_subset = random_split(full_val_source, [train_size, val_size], generator=generator)

    if config.get("quick_debug", False):
        train_subset = Subset(train_subset, range(min(int(config.get("debug_train_samples", 4096)), len(train_subset))))
        val_subset = Subset(val_subset, range(min(int(config.get("debug_val_samples", 1024)), len(val_subset))))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
