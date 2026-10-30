import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from dataloader import get_tiny_imagenet_dual_loaders
from student import DualReconstructionStudent

def get_param_groups(model, backbone_lr=2e-4, proj_lr=1e-3, head_lr=5e-5, weight_decay=1e-2):
    """
    Differential learning rates & weight decay:
      - Pretrained backbones (trunk, conv_island, dino_island): backbone_lr (2e-4) to preserve ImageNet filters
      - Reconstruction projectors (core_ln, conv_proj, recon_conv, dino_proj, recon_dino): proj_lr (1e-3) for fast alignment
      - Teacher-initialized probes (head_conv, head_dino): head_lr (5e-5) to stay anchored to teacher manifold (SimKD)
    """
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = (param.ndim < 2 or "bn" in name or "ln" in name or "norm" in name or "bias" in name)
        wd = 0.0 if is_no_decay else weight_decay

        if name.startswith(("trunk.", "conv_island.", "dino_island.")):
            lr = backbone_lr
            g_key = ("backbone", wd)
        elif name.startswith(("head_conv.", "head_dino.")):
            lr = head_lr
            g_key = ("head", wd)
        else:
            lr = proj_lr
            g_key = ("proj", wd)

        if g_key not in groups:
            groups[g_key] = {"params": [], "lr": lr, "weight_decay": wd}
        groups[g_key]["params"].append(param)

    return list(groups.values())

class GPUBatchTransform(nn.Module):
    """Ultra-fast GPU batch normalization and augmentation on native 64x64 tensors."""
    def __init__(self, device):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))
        self.train_aug = nn.Sequential(
            v2.RandomCrop(64, padding=4),
            v2.RandomHorizontalFlip(p=0.5),
        ).to(device)
        self.erasing = v2.RandomErasing(p=0.2, scale=(0.02, 0.15)).to(device)

    @torch.no_grad()
    def forward(self, x, train=True):
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        if train:
            x = self.train_aug(x)
        x = (x - self.mean) / self.std
        if train:
            x = self.erasing(x)
        return x

def train_one_epoch(epoch, epochs, student, loader, opt, scaler, gpu_transform, device, temp=2.0, alpha=0.7, beta=1.5, max_batches=None):
    student.train()
    total_loss, total_conv, total_dino = 0.0, 0.0, 0.0
    total_f1_loss, total_f2_loss = 0.0, 0.0
    corr_conv, corr_dino, corr_ens, total = 0, 0, 0, 0
    num_batches = max_batches if max_batches else len(loader)
    start_time = time.time()

    for i, (images, labels, t1_logits, t2_logits, t1_feat, t2_feat) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        images = gpu_transform(images.to(device, non_blocking=True), train=True)
        labels = labels.to(device, non_blocking=True)
        t1_logits = t1_logits.to(device, non_blocking=True)
        t2_logits = t2_logits.to(device, non_blocking=True)
        t1_feat = t1_feat.to(device, non_blocking=True)
        t2_feat = t2_feat.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            logits_conv, logits_dino, logits_ens, emb_conv, emb_dino, _ = student(images)

        # Enforce Float32 for numerical stability in loss computation
        l_conv_f32 = logits_conv.float()
        l_dino_f32 = logits_dino.float()
        t1_l_f32 = t1_logits.float()
        t2_l_f32 = t2_logits.float()
        emb_conv_f32 = emb_conv.float()
        emb_dino_f32 = emb_dino.float()
        t1_f_f32 = t1_feat.float()
        t2_f_f32 = t2_feat.float()

        # Stream 1: ConvNeXt Branch (1024d) — Direction (Cosine) + Magnitude (SmoothL1)
        ce_conv = F.cross_entropy(l_conv_f32, labels, label_smoothing=0.1)
        kd_conv = F.kl_div(
            F.log_softmax(l_conv_f32 / temp, dim=-1),
            F.softmax(t1_l_f32 / temp, dim=-1),
            reduction="batchmean"
        ) * (temp ** 2)
        cos_conv = (1.0 - F.cosine_similarity(emb_conv_f32, t1_f_f32, dim=-1)).mean()
        l1_conv = F.smooth_l1_loss(emb_conv_f32, t1_f_f32)
        feat_conv = cos_conv + 2.0 * l1_conv
        loss_conv = (1.0 - alpha) * ce_conv + alpha * kd_conv + beta * feat_conv

        # Stream 2: DINOv2 Branch (768d) — Direction (Cosine) + Magnitude (SmoothL1)
        ce_dino = F.cross_entropy(l_dino_f32, labels, label_smoothing=0.1)
        kd_dino = F.kl_div(
            F.log_softmax(l_dino_f32 / temp, dim=-1),
            F.softmax(t2_l_f32 / temp, dim=-1),
            reduction="batchmean"
        ) * (temp ** 2)
        cos_dino = (1.0 - F.cosine_similarity(emb_dino_f32, t2_f_f32, dim=-1)).mean()
        l1_dino = F.smooth_l1_loss(emb_dino_f32, t2_f_f32)
        feat_dino = cos_dino + 2.0 * l1_dino
        loss_dino = (1.0 - alpha) * ce_dino + alpha * kd_dino + beta * feat_dino

        # Total Dual-Stream Loss
        total_batch_loss = 0.5 * loss_conv + 0.5 * loss_dino

        opt.zero_grad(set_to_none=True)
        scaler.scale(total_batch_loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        scaler.step(opt)
        scaler.update()

        bs = images.size(0)
        total_loss += total_batch_loss.item() * bs
        total_conv += loss_conv.item() * bs
        total_dino += loss_dino.item() * bs
        total_f1_loss += cos_conv.item() * bs
        total_f2_loss += cos_dino.item() * bs

        corr_conv += (l_conv_f32.argmax(dim=-1) == labels).sum().item()
        corr_dino += (l_dino_f32.argmax(dim=-1) == labels).sum().item()
        corr_ens += (logits_ens.argmax(dim=-1) == labels).sum().item()
        total += bs

        if (i + 1) % 100 == 0 or (i + 1) == num_batches:
            elapsed = time.time() - start_time
            acc_c = corr_conv / total * 100
            acc_d = corr_dino / total * 100
            acc_e = corr_ens / total * 100
            print(f"Epoch [{epoch}/{epochs}] Step [{i+1}/{num_batches}] ({((i+1)/num_batches)*100:.1f}%) | "
                  f"Loss: {total_loss/total:.4f} (Conv: {total_conv/total:.3f} | DINO: {total_dino/total:.3f}) | "
                  f"CosDist1: {total_f1_loss/total:.3f} CosDist2: {total_f2_loss/total:.3f} | "
                  f"Acc: [Conv: {acc_c:.1f}% | DINO: {acc_d:.1f}% | Ens: {acc_e:.1f}%] | {elapsed:.1f}s")

    metrics = {
        "loss": total_loss / total,
        "acc_conv": corr_conv / total * 100,
        "acc_dino": corr_dino / total * 100,
        "acc_ens": corr_ens / total * 100,
    }
    return metrics

def evaluate(student, loader, joint_probe, gpu_transform, device, max_batches=None):
    student.eval()
    if joint_probe is not None:
        joint_probe.eval()

    corr_conv, corr_dino, corr_ens, corr_probe, total = 0, 0, 0, 0, 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            images = gpu_transform(images.to(device, non_blocking=True), train=False)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits_conv, logits_dino, logits_ens, _, _, emb_joint = student(images)
                if joint_probe is not None:
                    logits_probe = joint_probe(emb_joint)
                else:
                    logits_probe = logits_ens

            corr_conv += (logits_conv.argmax(dim=-1) == labels).sum().item()
            corr_dino += (logits_dino.argmax(dim=-1) == labels).sum().item()
            corr_ens += (logits_ens.argmax(dim=-1) == labels).sum().item()
            corr_probe += (logits_probe.argmax(dim=-1) == labels).sum().item()
            total += images.size(0)

    val_accs = {
        "conv": corr_conv / total * 100,
        "dino": corr_dino / total * 100,
        "ens": corr_ens / total * 100,
        "probe": corr_probe / total * 100
    }
    return val_accs

def main():
    parser = argparse.ArgumentParser(description="Dual-Stream Reconstructed Multi-Teacher Distillation (Option B)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--backbone-lr", type=float, default=2e-4, help="LR for pretrained backbones")
    parser.add_argument("--proj-lr", type=float, default=1e-3, help="LR for feature reconstruction projectors")
    parser.add_argument("--head-lr", type=float, default=5e-5, help="LR for teacher-initialized classifier heads")
    parser.add_argument("--l2", type=float, default=2e-2, help="Weight decay")
    parser.add_argument("--temp", type=float, default=2.0, help="KD distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight for soft teacher logit KD")
    parser.add_argument("--beta", type=float, default=1.5, help="Weight for teacher feature manifold alignment")
    parser.add_argument("--targets-path", type=str, default="tiny_dual_targets.pt", help="Path to dual teacher targets")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches per epoch for quick testing")
    parser.add_argument("--save-path", type=str, default="student_dual_option_b.pt", help="Checkpoint save path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print("=" * 78)
    print(" DUAL-STREAM RECONSTRUCTION DISTILLATION (SIMKD PROBE + FULL-BASIS 128x128)")
    print(f" Device: {device} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print(f" LRs -> Backbones: {args.backbone_lr} | Projectors: {args.proj_lr} | Probes: {args.head_lr}")
    print(f" Loss Weights -> Alpha(KD): {args.alpha} | Beta(Feat): {args.beta} | Temp: {args.temp}")
    print("=" * 78)

    if not os.path.exists(args.targets_path):
        print(f"\n[!] Targets not found at '{args.targets_path}'. Running 'python cache.py'...")
        from cache import main as build_targets
        build_targets()

    # Load targets and teacher probes
    data = torch.load(args.targets_path, map_location="cpu", weights_only=False)
    joint_probe = nn.Linear(1792, 200).to(device)
    joint_probe.load_state_dict(data["joint_probe_state"])

    train_loader, val_loader = get_tiny_imagenet_dual_loaders(targets_path=args.targets_path, batch_size=args.batch_size)
    print(f"Loaded {len(train_loader)} training batches in RAM with exact dual targets.\n")

    student = DualReconstructionStudent(num_classes=200, pretrained=True).to(device)

    # SimKD: Initialize student heads directly with the pre-fitted Teacher Probes!
    if "probe_conv_state" in data and "probe_dino_state" in data:
        student.head_conv.load_state_dict(data["probe_conv_state"])
        student.head_dino.load_state_dict(data["probe_dino_state"])
        print("[SimKD] Loaded pre-fitted Teacher Probes into student.head_conv (97.5%) & student.head_dino (97.0%).")

    gpu_transform = GPUBatchTransform(device)
    param_groups = get_param_groups(
        student,
        backbone_lr=args.backbone_lr,
        proj_lr=args.proj_lr,
        head_lr=args.head_lr,
        weight_decay=args.l2
    )
    opt = torch.optim.AdamW(param_groups, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")

    best_val_ens = 0.0
    for epoch in range(1, args.epochs + 1):
        cur_lr = opt.param_groups[0]["lr"]
        m = train_one_epoch(
            epoch, args.epochs, student, train_loader, opt, scaler, gpu_transform, device,
            temp=args.temp, alpha=args.alpha, beta=args.beta, max_batches=args.max_batches
        )
        scheduler.step()

        print("Evaluating validation set...")
        val = evaluate(student, val_loader, joint_probe, gpu_transform, device)
        print(f"\n>>> Epoch {epoch}/{args.epochs} SUMMARY: Loss: {m['loss']:.4f} | Backbone LR: {cur_lr:.6f}")
        print(f">>> Train Acc: [ConvNeXt: {m['acc_conv']:.2f}% | DINOv2: {m['acc_dino']:.2f}% | Ensemble: {m['acc_ens']:.2f}%]")
        print(f">>> Val Acc:   [ConvNeXt: {val['conv']:.2f}% | DINOv2: {val['dino']:.2f}% | Ensemble: {val['ens']:.2f}% | Joint Probe: {val['probe']:.2f}%]")

        if val["ens"] > best_val_ens:
            best_val_ens = val["ens"]
            torch.save({
                "student_state": student.state_dict(),
                "joint_probe_state": joint_probe.state_dict(),
                "epoch": epoch,
                "best_val_ens": best_val_ens,
                "best_val_probe": val["probe"]
            }, args.save_path)
            print(f">>> Saved Best Model to {args.save_path} (Val Ensemble: {best_val_ens:.2f}%)\n")

if __name__ == "__main__":
    main()
