import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_from_disk
import timm
from dataloader import TinyImageNetHFDataset

class DualBackbone(nn.Module):
    """Frozen Teacher Ensemble: ConvNeXt-Base (1024d) + DINOv2-ViT-Base (768d)."""
    def __init__(self):
        super().__init__()
        self.bb1 = timm.create_model("convnext_base", pretrained=True, num_classes=0)
        self.bb2 = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True
        )
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        x_224 = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        f1 = self.bb1(x_224)
        f2 = self.bb2(x_224)
        return f1, f2

def extract_teacher_features(features_cache_path, device, batch_size=128, data_path="tiny-imagenet-200-clean/tiny-imagenet-200-clean"):
    print("=" * 70)
    print(f" EXTRACTING RAW TEACHER FEATURES -> {features_cache_path}")
    print("=" * 70)
    transform_clean = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ds = load_from_disk(data_path)
    train_set = TinyImageNetHFDataset(ds["train"], transform=transform_clean)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    teachers = DualBackbone().to(device)
    teachers.eval()

    f1_list, f2_list, label_list = [], [], []
    start_time = time.time()
    num_batches = len(loader)

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                f1, f2 = teachers(images)
            f1_list.append(f1.half().cpu())
            f2_list.append(f2.half().cpu())
            label_list.append(labels.cpu())

            if (i + 1) % 50 == 0 or (i + 1) == num_batches:
                elapsed = time.time() - start_time
                print(f"  Cached [{i+1}/{num_batches}] batches ({((i+1)/num_batches)*100:.1f}%) | {elapsed:.1f}s")

    F1 = torch.cat(f1_list, dim=0)
    F2 = torch.cat(f2_list, dim=0)
    all_labels = torch.cat(label_list, dim=0)

    torch.save({"F1": F1, "F2": F2, "labels": all_labels}, features_cache_path)
    print(f"Saved raw teacher features to '{features_cache_path}' ({os.path.getsize(features_cache_path)/(1024*1024):.2f} MB)\n")
    del teachers
    torch.cuda.empty_cache()

def fit_probe(features, labels, device, epochs=6, batch_size=512, lr=3e-3):
    in_dim = features.shape[1]
    probe = nn.Linear(in_dim, 200).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-3)
    dataset = torch.utils.data.TensorDataset(features, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    for ep in range(1, epochs + 1):
        corr, tot = 0, 0
        for x_b, y_b in loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            out = probe(x_b)
            loss = F.cross_entropy(out, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            corr += (out.argmax(dim=-1) == y_b).sum().item()
            tot += y_b.size(0)
    print(f"  Fitted Probe ({in_dim}d -> 200) Final Training Accuracy: {corr/tot*100:.2f}%")
    return probe

def generate_logits(probe, features, device, batch_size=1024):
    probe.eval()
    dataset = torch.utils.data.TensorDataset(features)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logits_list = []
    with torch.no_grad():
        for (x_b,) in loader:
            x_b = x_b.to(device)
            logits_list.append(probe(x_b).half().cpu())
    return torch.cat(logits_list, dim=0)

def main():
    features_cache_path = "tiny_features_cache_convnext_dinov2.pt"
    save_path = "tiny_dual_targets.pt"

    print("=" * 70)
    print(" PRECOMPUTING DUAL TEACHER PROBES & TARGETS (OPTION B)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(features_cache_path):
        print(f"[!] '{features_cache_path}' not found. Extracting teacher features first...")
        extract_teacher_features(features_cache_path, device)

    print(f"Device: {device} | Loading {features_cache_path}...")
    data = torch.load(features_cache_path, map_location="cpu", weights_only=False)
    F1 = data["F1"].float()  # [98179, 1024] ConvNeXt-Base
    F2 = data["F2"].float()  # [98179, 768]  DINOv2-ViT-Base
    labels = data["labels"]  # [98179]

    # 1. Fit Probe 1 on ConvNeXt-Base (1024d)
    print("\n1. Fitting ConvNeXt-Base Teacher Probe (1024d -> 200)...")
    probe_conv = fit_probe(F1, labels, device)
    t1_logits = generate_logits(probe_conv, F1, device)

    # 2. Fit Probe 2 on DINOv2-ViT-Base (768d)
    print("\n2. Fitting DINOv2-ViT Teacher Probe (768d -> 200)...")
    probe_dino = fit_probe(F2, labels, device)
    t2_logits = generate_logits(probe_dino, F2, device)

    # 3. Fit Joint Teacher Ensemble Probe (1792d)
    print("\n3. Fitting Joint Teacher Ensemble Probe (1792d -> 200)...")
    F_joint = torch.cat([F1, F2], dim=1)
    probe_joint = fit_probe(F_joint, labels, device)
    t_joint_logits = generate_logits(probe_joint, F_joint, device)

    dual_targets = {
        "F1": F1.half(),                         # [98179, 1024]
        "F2": F2.half(),                         # [98179, 768]
        "t1_logits": t1_logits,                  # [98179, 200]
        "t2_logits": t2_logits,                  # [98179, 200]
        "t_joint_logits": t_joint_logits,        # [98179, 200]
        "joint_probe_state": probe_joint.state_dict(),
        "probe_conv_state": probe_conv.state_dict(),
        "probe_dino_state": probe_dino.state_dict(),
    }
    torch.save(dual_targets, save_path)
    print(f"\nSaved all dual targets to '{save_path}' ({os.path.getsize(save_path)/(1024*1024):.2f} MB)")
    print("=" * 70)

if __name__ == "__main__":
    main()
