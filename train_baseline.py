import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import get_tiny_imagenet_distill_loaders
from student import ModularStudent

class LogitKD(nn.Module):
    def __init__(self, temp=4.0, alpha=0.5, use_l2=False):
        super().__init__()
        self.temp = temp
        self.alpha = alpha
        self.use_l2 = use_l2

    def forward(self, student_logits, teacher_logits, labels):
        ce_loss = F.cross_entropy(student_logits, labels)
        if self.use_l2:
            kd_loss = F.mse_loss(student_logits, teacher_logits)
        else:
            p_s = F.log_softmax(student_logits / self.temp, dim=-1)
            p_t = F.softmax(teacher_logits / self.temp, dim=-1)
            kd_loss = F.kl_div(p_s, p_t, reduction="batchmean") * (self.temp ** 2)
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
        return total_loss, ce_loss, kd_loss

def train_one_epoch(epoch, epochs, student, loader, opt_s, criterion, scaler, device, max_batches=None):
    student.train()
    total_loss, total_ce, total_kd, correct, total = 0.0, 0.0, 0.0, 0, 0
    alpha_vals = []
    num_batches = max_batches if max_batches else len(loader)
    start_time = time.time()

    for i, (images, labels, t_logits) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        t_logits = t_logits.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            # Fast GPU interpolation: 64x64 -> 224x224 in 0.8ms
            images_224 = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
            student_logits, _, _, _, alpha = student(images_224)
            loss_s, ce_loss, kd_loss = criterion(student_logits, t_logits, labels)

        opt_s.zero_grad()
        scaler.scale(loss_s).backward()
        scaler.step(opt_s)
        scaler.update()

        total_loss += loss_s.item() * images.size(0)
        total_ce += ce_loss.item() * images.size(0)
        total_kd += kd_loss.item() * images.size(0)
        correct += (student_logits.argmax(dim=-1) == labels).sum().item()
        total += images.size(0)
        alpha_vals.append(alpha.detach().mean(dim=0))

        if (i + 1) % 100 == 0 or (i + 1) == num_batches:
            elapsed = time.time() - start_time
            cur_acc = correct / total * 100
            cur_loss = total_loss / total
            cur_ce = total_ce / total
            cur_kd = total_kd / total
            cur_a = alpha.detach().mean(dim=0)
            print(f"Epoch [{epoch}/{epochs}] Step [{i+1}/{num_batches}] ({((i+1)/num_batches)*100:.1f}%) | "
                  f"Loss: {cur_loss:.4f} (CE: {cur_ce:.4f} | KD: {cur_kd:.4f}) | Acc: {cur_acc:.1f}% | "
                  f"Sat1: {cur_a[0]:.3f} Sat2: {cur_a[1]:.3f} | {elapsed:.1f}s")

    avg_alpha = torch.stack(alpha_vals).mean(dim=0)
    return total_loss / total, total_ce / total, total_kd / total, correct / total * 100, avg_alpha

def evaluate(student, loader, device, max_batches=None):
    student.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                images_224 = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
                logits, _, _, _, _ = student(images_224)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += images.size(0)
    return correct / total * 100

def main():
    parser = argparse.ArgumentParser(description="Blazing-Fast Pre-GNN Distillation on Tiny-ImageNet-200")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--l2", "--weight-decay", dest="l2", type=float, default=1e-2, help="L2 regularization / weight decay")
    parser.add_argument("--use-l2-kd", action="store_true", help="Use L2 MSE loss for KD instead of KL divergence")
    parser.add_argument("--targets-path", type=str, default="tiny_teacher_targets.pt", help="Path to precomputed teacher targets")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches per epoch (for quick testing)")
    parser.add_argument("--save-path", type=str, default="student_baseline.pt", help="Checkpoint save path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Blazing-Fast Pre-GNN Distillation on Tiny-ImageNet-200")
    print(f"Device: {device} | Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr} | L2/WD: {args.l2}")

    if not os.path.exists(args.targets_path):
        print(f"\n[!] Precomputed targets not found at '{args.targets_path}'.")
        print("    Run 'python cache_teacher_targets.py' first (takes ~2.5 mins once)!")
        return

    train_loader, val_loader = get_tiny_imagenet_distill_loaders(targets_path=args.targets_path, batch_size=args.batch_size)
    print(f"Loaded {len(train_loader)} training batches with precomputed teacher graph targets.\n")

    student = ModularStudent(core_dim=128, sat1_dim=32, sat2_dim=16, num_classes=200).to(device)
    opt_s = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    criterion = LogitKD(temp=4.0, alpha=0.5, use_l2=args.use_l2_kd)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        loss, ce, kd, train_acc, avg_alpha = train_one_epoch(
            epoch, args.epochs, student, train_loader, opt_s, criterion, scaler, device, max_batches=args.max_batches
        )
        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]

        print("Evaluating validation set...")
        val_acc = evaluate(student, val_loader, device)
        print(f"\n>>> Epoch {epoch}/{args.epochs} SUMMARY: Loss: {loss:.4f} (CE: {ce:.4f} | KD: {kd:.4f}) | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | LR: {cur_lr:.6f}")
        print(f">>> Dynamic Gating: Sat1 (ResNet bias) = {avg_alpha[0]:.3f} | Sat2 (Swin bias) = {avg_alpha[1]:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), args.save_path)
            print(f">>> Saved Best Model to {args.save_path} (Val Acc: {best_val_acc:.2f}%)\n")

if __name__ == "__main__":
    main()
