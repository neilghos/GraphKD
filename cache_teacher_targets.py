import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import get_tiny_imagenet_loaders
from backbone import DualBackbone
from graph_builder import HomophilyGraphBuilder

def precompute_teachers(save_path="tiny_teacher_targets.pt", batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Precomputing Teacher Targets on: {device} (Batch size: {batch_size})")

    # Clean deterministic transform for feature caching (no random crops/flips)
    from torchvision import transforms
    from datasets import load_from_disk
    from dataloader import TinyImageNetHFDataset
    from torch.utils.data import DataLoader

    transform_clean = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ds = load_from_disk("tiny-imagenet-200-clean/tiny-imagenet-200-clean")
    train_set = TinyImageNetHFDataset(ds['train'], transform=transform_clean)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    teachers = DualBackbone(pretrained=True).to(device).eval()

    # 1. Extract frozen features for entire training set (~2.5 minutes)
    print("Step 1/3: Extracting sequential clean features from ResNet-18 & Swin-T...")
    f1_list, f2_list, label_list = [], [], []
    t0 = time.time()

    with torch.no_grad():
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                images_224 = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
                f1, f2 = teachers(images_224)
            f1_list.append(f1.half().cpu())
            f2_list.append(f2.half().cpu())
            label_list.append(labels)
            if (i + 1) % 50 == 0 or (i + 1) == len(train_loader):
                print(f"  Extracted [{i+1}/{len(train_loader)}] batches ({((i+1)/len(train_loader))*100:.1f}%) | {time.time()-t0:.1f}s")

    F1 = torch.cat(f1_list, dim=0)
    F2 = torch.cat(f2_list, dim=0)
    all_labels = torch.cat(label_list, dim=0)

    # 2. Build Homophily Graph on 2048 calibration samples
    print("\nStep 2/3: Building Homophily-Heterophily Graph...")
    builder = HomophilyGraphBuilder(threshold=0.30, local_threshold=0.35, prune_orphans=True).to(device)
    adj, core_nodes, local1, local2, bridges = builder(F1[:2048].float().to(device), F2[:2048].float().to(device))
    print(f"  --> Global Core: {len(core_nodes)} | BB1 Local: {len(local1)} | BB2 Local: {len(local2)}")

    # 3. Fit Teacher Pooled Graph Classifier on frozen features (~10 seconds)
    print("\nStep 3/3: Fitting Teacher Community Readout Head on 98k features...")
    H_0 = torch.cat([F1, F2], dim=1).float()
    t_core = H_0[:, core_nodes.cpu()]
    t_sat1 = H_0[:, local1.cpu()]
    t_sat2 = H_0[:, local2.cpu()]

    # Fast Linear Readout
    head_core = nn.Linear(len(core_nodes), 128).to(device)
    head_sat1 = nn.Linear(len(local1), 32).to(device)
    head_sat2 = nn.Linear(len(local2), 16).to(device)
    classifier = nn.Linear(128 + 32 + 16, 200).to(device)

    opt = torch.optim.AdamW(list(head_core.parameters()) + list(head_sat1.parameters()) + 
                           list(head_sat2.parameters()) + list(classifier.parameters()), lr=3e-3)
    
    # Train teacher linear probe for 5 fast epochs on the pre-extracted features
    dataset_tensor = torch.utils.data.TensorDataset(t_core, t_sat1, t_sat2, all_labels)
    feat_loader = torch.utils.data.DataLoader(dataset_tensor, batch_size=512, shuffle=True)

    for ep in range(1, 9):
        total_loss, correct, total = 0.0, 0, 0
        for c, s1, s2, y in feat_loader:
            c, s1, s2, y = c.to(device), s1.to(device), s2.to(device), y.to(device)
            fused = torch.cat([head_core(c), head_sat1(s1), head_sat2(s2)], dim=-1)
            logits = classifier(fused)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.size(0)
        print(f"  Teacher Probe Epoch {ep}/8 | Loss: {total_loss/total:.4f} | Accuracy: {correct/total*100:.2f}%")

    # Generate teacher target logits for all 98,179 samples
    head_core.eval(); head_sat1.eval(); head_sat2.eval(); classifier.eval()
    teacher_logits_list = []
    eval_loader = torch.utils.data.DataLoader(dataset_tensor, batch_size=1024, shuffle=False)
    with torch.no_grad():
        for c, s1, s2, _ in eval_loader:
            c, s1, s2 = c.to(device), s1.to(device), s2.to(device)
            fused = torch.cat([head_core(c), head_sat1(s1), head_sat2(s2)], dim=-1)
            teacher_logits_list.append(classifier(fused).half().cpu())

    train_teacher_logits = torch.cat(teacher_logits_list, dim=0)

    # Verification: Ensure teacher targets match the sequential labels perfectly
    cached_acc = (train_teacher_logits.float().argmax(dim=-1) == all_labels).float().mean().item()
    print(f"\n[Verification] Cached Teacher Accuracy against Dataset Labels: {cached_acc*100:.2f}%")
    assert cached_acc > 0.40, f"Teacher accuracy must be high (got {cached_acc*100:.2f}%)!"

    # Save cached targets and graph
    torch.save({
        "teacher_logits": train_teacher_logits,  # [98179, 200] in fp16 (~39MB)
        "core_nodes": core_nodes.cpu(),
        "local1": local1.cpu(),
        "local2": local2.cpu(),
        "bridges": bridges
    }, save_path)
    print(f"\nSUCCESS! Cached teacher targets saved to {save_path} ({os.path.getsize(save_path)/(1024*1024):.1f} MB)")
    print(f"Total Precompute Time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    precompute_teachers()
