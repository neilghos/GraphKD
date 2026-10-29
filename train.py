import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import get_tiny_imagenet_dual_loaders
from student_dual import DualReconstructionStudent

def train_one_epoch(epoch, epochs, student, loader, opt, scaler, device, temp=2.0, alpha=0.5, beta=0.2, max_batches=None):
    student.train()
    total_loss, total_conv, total_dino = 0.0, 0.0, 0.0
    total_f1_loss, total_f2_loss = 0.0, 0.0
    corr_conv, corr_dino, corr_ens, total = 0, 0, 0, 0
    num_batches = max_batches if max_batches else len(loader)
    start_time = time.time()

    for i, (images, labels, t1_logits, t2_logits, t1_feat, t2_feat) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        t1_logits = t1_logits.to(device, non_blocking=True)
        t2_logits = t2_logits.to(device, non_blocking=True)
        t1_feat = t1_feat.to(device, non_blocking=True)
        t2_feat = t2_feat.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            images_224 = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
            logits_conv, logits_dino, logits_ens, emb_conv, emb_dino, _ = student(images_224)

        # Enforce Float32 for numerical stability in loss computation
        l_conv_f32 = logits_conv.float()
        l_dino_f32 = logits_dino.float()
        t1_l_f32 = t1_logits.float()
        t2_l_f32 = t2_logits.float()

        # Stream 1: ConvNeXt Branch (1024d)
        ce_conv = F.cross_entropy(l_conv_f32, labels)
        kd_conv = F.kl_div(
            F.log_softmax(l_conv_f32 / temp, dim=-1),
            F.softmax(t1_l_f32 / temp, dim=-1),
            reduction="batchmean"
        ) * (temp ** 2)
        feat_conv = (1.0 - F.cosine_similarity(emb_conv.float(), t1_feat.float(), dim=-1)).mean()
        loss_conv = (1.0 - alpha) * ce_conv + alpha * kd_conv + beta * feat_conv

        # Stream 2: DINOv2 Branch (768d)
        ce_dino = F.cross_entropy(l_dino_f32, labels)
        kd_dino = F.kl_div(
            F.log_softmax(l_dino_f32 / temp, dim=-1),
            F.softmax(t2_l_f32 / temp, dim=-1),
            reduction="batchmean"
        ) * (temp ** 2)
        feat_dino = (1.0 - F.cosine_similarity(emb_dino.float(), t2_feat.float(), dim=-1)).mean()
        loss_dino = (1.0 - alpha) * ce_dino + alpha * kd_dino + beta * feat_dino

        # Total Dual-Stream Loss
        total_batch_loss = 0.5 * loss_conv + 0.5 * loss_dino

        opt.zero_grad()
        scaler.scale(total_batch_loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        scaler.step(opt)
        scaler.update()

        bs = images.size(0)
        total_loss += total_batch_loss.item() * bs
        total_conv += loss_conv.item() * bs
        total_dino += loss_dino.item() * bs
        total_f1_loss += feat_conv.item() * bs
        total_f2_loss += feat_dino.item() * bs

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
                  f"Feat1: {total_f1_loss/total:.3f} Feat2: {total_f2_loss/total:.3f} | "
                  f"Acc: [Conv: {acc_c:.1f}% | DINO: {acc_d:.1f}% | Ens: {acc_e:.1f}%] | {elapsed:.1f}s")

    metrics = {
        "loss": total_loss / total,
        "acc_conv": corr_conv / total * 100,
        "acc_dino": corr_dino / total * 100,
        "acc_ens": corr_ens / total * 100,
    }
    return metrics

def evaluate(student, loader, joint_probe, device, max_batches=None):
    student.eval()
    if joint_probe is not None:
        joint_probe.eval()

    corr_conv, corr_dino, corr_ens, corr_probe, total = 0, 0, 0, 0, 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                images_224 = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
                logits_conv, logits_dino, logits_ens, _, _, emb_joint = student(images_224)
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
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--l2", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--temp", type=float, default=2.0, help="KD distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for logit KD")
    parser.add_argument("--beta", type=float, default=0.2, help="Weight for direct feature cosine matching")
    parser.add_argument("--targets-path", type=str, default="tiny_dual_targets.pt", help="Path to dual teacher targets")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches per epoch for quick testing")
    parser.add_argument("--save-path", type=str, default="student_dual_option_b.pt", help="Checkpoint save path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(" DUAL-STREAM EXACT RECONSTRUCTION DISTILLATION (OPTION B)")
    print(f" Device: {device} | Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f" Loss Weights: Alpha(KD): {args.alpha} | Beta(Feature Matching): {args.beta} | Temp: {args.temp}")
    print("=" * 75)

    if not os.path.exists(args.targets_path):
        print(f"\n[!] Targets not found at '{args.targets_path}'. Running 'python cache_dual_targets.py'...")
        from cache_dual_targets import main as build_targets
        build_targets()

    # Load targets and joint probe
    data = torch.load(args.targets_path, map_location="cpu", weights_only=False)
    joint_probe = nn.Linear(1792, 200).to(device)
    joint_probe.load_state_dict(data["joint_probe_state"])

    train_loader, val_loader = get_tiny_imagenet_dual_loaders(targets_path=args.targets_path, batch_size=args.batch_size)
    print(f"Loaded {len(train_loader)} training batches with exact dual targets.\n")

    student = DualReconstructionStudent(num_classes=200, pretrained=True).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")

    best_val_probe = 0.0
    for epoch in range(1, args.epochs + 1):
        m = train_one_epoch(
            epoch, args.epochs, student, train_loader, opt, scaler, device,
            temp=args.temp, alpha=args.alpha, beta=args.beta, max_batches=args.max_batches
        )
        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]

        print("Evaluating validation set...")
        val = evaluate(student, val_loader, joint_probe, device)
        print(f"\n>>> Epoch {epoch}/{args.epochs} SUMMARY: Loss: {m['loss']:.4f} | LR: {cur_lr:.6f}")
        print(f">>> Train Acc: [ConvNeXt: {m['acc_conv']:.2f}% | DINOv2: {m['acc_dino']:.2f}% | Ensemble: {m['acc_ens']:.2f}%]")
        print(f">>> Val Acc:   [ConvNeXt: {val['conv']:.2f}% | DINOv2: {val['dino']:.2f}% | Ensemble: {val['ens']:.2f}% | Joint Probe: {val['probe']:.2f}%]")

        if val["probe"] > best_val_probe:
            best_val_probe = val["probe"]
            torch.save({
                "student_state": student.state_dict(),
                "joint_probe_state": joint_probe.state_dict(),
                "epoch": epoch,
                "best_val_probe": best_val_probe
            }, args.save_path)
            print(f">>> Saved Best Model to {args.save_path} (Val Joint Probe: {best_val_probe:.2f}%)\n")

if __name__ == "__main__":
    main()
