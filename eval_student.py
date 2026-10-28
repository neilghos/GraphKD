import time
import torch
import torch.nn.functional as F
from dataloader import get_tiny_imagenet_loaders
from student import ModularStudent

def evaluate_split(split_name="test", checkpoint_path="student_baseline.pt", head_type=None, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-detect head_type if not specified
    if head_type is None:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        head_type = "inductive" if "sat1_conv.0.weight" in state_dict else "mlp"
    else:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    print("=" * 70)
    print(f" EVALUATION ON SPLIT: [{split_name.upper()}] (Head Type: {head_type})")
    print(f" Device: {device} | Checkpoint: {checkpoint_path}")
    print("=" * 70)

    # 1. Load Data
    from torchvision import transforms
    from datasets import load_from_disk
    from dataloader import TinyImageNetHFDataset
    from torch.utils.data import DataLoader

    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ds = load_from_disk("tiny-imagenet-200-clean/tiny-imagenet-200-clean")
    eval_set = TinyImageNetHFDataset(ds[split_name], transform=transform_eval)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    total_samples = len(eval_loader.dataset)
    print(f"Loaded {split_name} split: {total_samples} images across 200 classes.")

    # 2. Load Model
    student = ModularStudent(core_dim=128, sat1_dim=32, sat2_dim=16, num_classes=200, head_type=head_type).to(device)
    student.load_state_dict(state_dict)
    student.eval()

    # Parameter counts
    total_params = sum(p.numel() for p in student.parameters())
    trunk_params = sum(p.numel() for p in student.trunk.parameters())
    core_params = sum(p.numel() for p in student.core_head.parameters())
    if hasattr(student, 'sat1_head'):
        sat1_params = sum(p.numel() for p in student.sat1_head.parameters())
        sat2_params = sum(p.numel() for p in student.sat2_head.parameters())
    else:
        sat1_params = sum(p.numel() for p in student.sat1_conv.parameters()) + sum(p.numel() for p in student.sat1_proj.parameters())
        sat2_params = (sum(p.numel() for p in student.sat2_in_proj.parameters()) +
                       sum(p.numel() for p in student.sat2_mha.parameters()) +
                       sum(p.numel() for p in student.sat2_ln.parameters()) +
                       sum(p.numel() for p in student.sat2_proj.parameters()))
    gate_params = sum(p.numel() for p in student.gate.parameters())
    cls_params = sum(p.numel() for p in student.classifier.parameters())

    print(f"\n--- Architecture & Parameter Breakdown ---")
    print(f"  MobileNetV3-Small Trunk : {trunk_params:>10,} params ({trunk_params/total_params*100:4.1f}%)")
    print(f"  Core Invariant Head     : {core_params:>10,} params ({core_params/total_params*100:4.1f}%)")
    print(f"  Sat 1 (ResNet Specialist): {sat1_params:>10,} params ({sat1_params/total_params*100:4.1f}%)")
    print(f"  Sat 2 (Swin Specialist) : {sat2_params:>10,} params ({sat2_params/total_params*100:4.1f}%)")
    print(f"  Dynamic Router Gate     : {gate_params:>10,} params ({gate_params/total_params*100:4.1f}%)")
    print(f"  Fused Classifier Head   : {cls_params:>10,} params ({cls_params/total_params*100:4.1f}%)")
    print(f"  TOTAL STUDENT PARAMS    : {total_params:>10,} params (~{total_params/1e6:.2f}M)")

    # 3. Evaluation Loop
    top1_correct = 0
    top5_correct = 0
    total = 0
    all_alphas = []

    # Warmup GPU
    dummy = torch.randn(16, 3, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(5):
            _ = student(dummy)

    start_eval_time = time.time()
    with torch.no_grad():
        for images, labels in eval_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                images_224 = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
                logits, s_core, s_sat1, s_sat2, alpha = student(images_224)

            # Top-1 & Top-5 accuracy
            _, pred = logits.topk(5, dim=-1, largest=True, sorted=True)
            correct_matrix = pred.eq(labels.view(-1, 1).expand_as(pred))
            
            top1_correct += correct_matrix[:, 0].sum().item()
            top5_correct += correct_matrix[:, :5].sum().item()
            total += labels.size(0)

            all_alphas.append(alpha.detach().cpu())

    eval_duration = time.time() - start_eval_time
    latency_per_sample_ms = (eval_duration / total) * 1000
    fps = total / eval_duration

    # Aggregate Gating
    all_alphas = torch.cat(all_alphas, dim=0)  # [N, 2]
    sat1_vals = all_alphas[:, 0]
    sat2_vals = all_alphas[:, 1]

    swin_preferred = (sat2_vals > sat1_vals).float().mean().item() * 100
    resnet_preferred = (sat1_vals > sat2_vals).float().mean().item() * 100

    top1_acc = (top1_correct / total) * 100
    top5_acc = (top5_correct / total) * 100

    print("\n" + "=" * 70)
    print(f" FINAL [{split_name.upper()}] BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  Top-1 Accuracy       : {top1_acc:.2f}%")
    print(f"  Top-5 Accuracy       : {top5_acc:.2f}%")
    print(f"  Inference Latency    : {latency_per_sample_ms:.2f} ms / image ({fps:.1f} FPS on {device})")

    print("\n--- Dynamic Gating Router Analysis ---")
    print(f"  Sat 1 (ResNet-18 Prior) Mean : {sat1_vals.mean().item():.4f} (std: {sat1_vals.std().item():.4f})")
    print(f"  Sat 2 (Swin-T Prior)    Mean : {sat2_vals.mean().item():.4f} (std: {sat2_vals.std().item():.4f})")
    print(f"  Swin-T Preferred Images      : {swin_preferred:.2f}%")
    print(f"  ResNet-18 Preferred Images   : {resnet_preferred:.2f}%")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Student Checkpoint on Validation and Test splits")
    parser.add_argument("checkpoint", nargs="?", default="student_baseline.pt", help="Path to checkpoint file")
    parser.add_argument("--head-type", type=str, default=None, choices=["mlp", "inductive"], help="Head type (auto-detected if None)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    args = parser.parse_args()

    evaluate_split("validation", checkpoint_path=args.checkpoint, head_type=args.head_type, batch_size=args.batch_size)
    evaluate_split("test", checkpoint_path=args.checkpoint, head_type=args.head_type, batch_size=args.batch_size)
