import os
import time
import numpy as np
import torch
from torchvision import transforms
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

def load_or_build_uint8_cache(path="tiny-imagenet-200-clean/tiny-imagenet-200-clean", cache_file="tiny_images_uint8.pt"):
    """Pre-packs all 64x64 RGB images into contiguous uint8 tensors in RAM (~1.25 GB total)."""
    if os.path.exists(cache_file):
        return torch.load(cache_file, map_location="cpu", weights_only=False)

    print(f"[Dataloader] Building one-time fast RAM image cache '{cache_file}'...")
    t0 = time.time()
    ds = load_from_disk(path)
    cache = {}
    for split in ["train", "validation", "test"]:
        if split not in ds:
            continue
        split_ds = ds[split]
        n = len(split_ds)
        imgs = torch.empty((n, 3, 64, 64), dtype=torch.uint8)
        labels = torch.tensor(split_ds["label"], dtype=torch.long)
        for i, img in enumerate(split_ds["image"]):
            arr = np.array(img.convert("RGB"), dtype=np.uint8, copy=True)
            imgs[i] = torch.from_numpy(arr).permute(2, 0, 1)
        cache[split] = {"images": imgs, "labels": labels}
        print(f"  Packed '{split}': {n} images ({imgs.numel() / (1024*1024):.1f} MB)")

    torch.save(cache, cache_file)
    print(f"[Dataloader] Saved RAM cache to '{cache_file}' in {time.time() - t0:.1f}s.\n")
    return cache

class FastTensorDualDataset(Dataset):
    """Zero-overhead in-RAM dataset yielding (uint8_img, label, t1_logits, t2_logits, t1_feat, t2_feat)."""
    def __init__(self, images_uint8, labels, t1_logits, t2_logits, t1_feat, t2_feat):
        self.images = images_uint8
        self.labels = labels
        self.t1_logits = t1_logits
        self.t2_logits = t2_logits
        self.t1_feat = t1_feat
        self.t2_feat = t2_feat

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        return (
            self.images[idx],
            self.labels[idx],
            self.t1_logits[idx],
            self.t2_logits[idx],
            self.t1_feat[idx],
            self.t2_feat[idx]
        )

class FastTensorEvalDataset(Dataset):
    """Zero-overhead in-RAM evaluation dataset yielding (uint8_img, label)."""
    def __init__(self, images_uint8, labels):
        self.images = images_uint8
        self.labels = labels

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

class FastBatchLoader:
    """
    C++-vectorized batch loader over contiguous RAM tensors.
    Eliminates 98,179 Python __getitem__ calls and 4,608 torch.stack collations per epoch.
    """
    def __init__(self, *tensors, batch_size=128, shuffle=True):
        self.tensors = tensors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = tensors[0].size(0)
        self.dataset = tensors[0]

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(self.n)
            for start in range(0, self.n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                yield tuple(t[idx] for t in self.tensors)
        else:
            for start in range(0, self.n, self.batch_size):
                end = min(start + self.batch_size, self.n)
                yield tuple(t[start:end] for t in self.tensors)

def get_tiny_imagenet_dual_loaders(targets_path="tiny_dual_targets.pt", batch_size=128, path="tiny-imagenet-200-clean/tiny-imagenet-200-clean"):
    if not os.path.exists(targets_path):
        raise FileNotFoundError(f"Targets path '{targets_path}' not found. Run python cache.py first!")

    img_cache = load_or_build_uint8_cache(path=path)
    data = torch.load(targets_path, map_location="cpu", weights_only=False)

    train_loader = FastBatchLoader(
        img_cache["train"]["images"],
        img_cache["train"]["labels"],
        data["t1_logits"],
        data["t2_logits"],
        data["F1"],
        data["F2"],
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = FastBatchLoader(
        img_cache["validation"]["images"],
        img_cache["validation"]["labels"],
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, val_loader
