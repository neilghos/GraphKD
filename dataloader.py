import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk

class TinyImageNetHFDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item['image'].convert('RGB')
        label = item['label']
        if self.transform:
            img = self.transform(img)
        return img, label

class TinyImageNetDistillDataset(Dataset):
    """Dataset yielding (image, label, teacher_logit) with zero teacher forward overhead"""
    def __init__(self, hf_dataset, teacher_logits, transform=None):
        self.ds = hf_dataset
        self.teacher_logits = teacher_logits
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item['image'].convert('RGB')
        label = item['label']
        if self.transform:
            img = self.transform(img)
        t_logit = self.teacher_logits[idx].float()
        return img, label, t_logit

def get_tiny_imagenet_distill_loaders(targets_path="tiny_teacher_targets.pt", batch_size=128, path="tiny-imagenet-200-clean/tiny-imagenet-200-clean"):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ds = load_from_disk(path)
    val_set = TinyImageNetHFDataset(ds['validation'], transform=transform_val)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    if os.path.exists(targets_path):
        data = torch.load(targets_path, weights_only=False)
        train_set = TinyImageNetDistillDataset(ds['train'], data["teacher_logits"], transform=transform_train)
    else:
        train_set = TinyImageNetHFDataset(ds['train'], transform=transform_train)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    return train_loader, val_loader

def get_tiny_imagenet_loaders(batch_size=128, path="tiny-imagenet-200-clean/tiny-imagenet-200-clean", num_workers=0):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ds = load_from_disk(path)
    train_set = TinyImageNetHFDataset(ds['train'], transform=transform_train)
    val_set = TinyImageNetHFDataset(ds['validation'], transform=transform_val)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

def get_cifar10_loaders(batch_size=64, img_size=224, root="./data"):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    train_set = datasets.CIFAR10(root=root, train=True, download=False, transform=transform)
    test_set = datasets.CIFAR10(root=root, train=False, download=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader
