import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import load_or_build_uint8_cache, FastTensorEvalDataset
from student import DualReconstructionStudent

def evaluate_split(split_name, checkpoint_path="student_dual_option_b.pt", batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(f" EVALUATION ON SPLIT: [{split_name.upper()}] (Option B: Native 64x64 Dual-Stream)")
    print(f" Device: {device} | Checkpoint: {checkpoint_path}")
    print("=" * 75)

    # 1. Load Data from RAM Cache
    img_cache = load_or_build_uint8_cache()
    eval_set = FastTensorEvalDataset(img_cache[split_name]["images"], img_cache[split_name]["labels"])
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    total_samples = len(eval_loader.dataset)
    print(f"Loaded {split_name} split: {total_samples} images across 200 classes.")

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # 2. Load Models
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    student = DualReconstructionStudent(num_classes=200, pretrained=False).to(device)
    student.load_state_dict(ckpt["student_state"])
    student.eval()

    joint_probe = nn.Linear(1792, 200).to(device)
    if "joint_probe_state" in ckpt:
        joint_probe.load_state_dict(ckpt["joint_probe_state"])
    joint_probe.eval()

    total_params = sum(p.numel() for p in student.parameters())
    trunk_params = sum(p.numel() for p in student.trunk.parameters()) + sum(p.numel() for p in student.core_ln.parameters())
    conv_spec_params = (sum(p.numel() for p in student.conv_island.parameters()) +
                        sum(p.numel() for p in student.conv_proj.parameters()) +
                        sum(p.numel() for p in student.recon_conv.parameters()) +
                        sum(p.numel() for p in student.head_conv.parameters()))
    dino_spec_params = (sum(p.numel() for p in student.dino_island.parameters()) +
                        sum(p.numel() for p in student.dino_proj.parameters()) +
                        sum(p.numel() for p in student.recon_dino.parameters()) +
                        sum(p.numel() for p in student.head_dino.parameters()))

    print(f"\n--- Architecture & Parameter Breakdown ---")
    print(f"  Shared Femto Trunk      : {trunk_params:>10,} params ({trunk_params/total_params*100:4.1f}%)")
    print(f"  ConvNeXt Stream (1024d) : {conv_spec_params:>10,} params ({conv_spec_params/total_params*100:4.1f}%)")
    print(f"  DINOv2 Stream (768d)    : {dino_spec_params:>10,} params ({dino_spec_params/total_params*100:4.1f}%)")
    print(f"  TOTAL STUDENT PARAMS    : {total_params:>10,} params (~{total_params/1e6:.2f}M)")

    # 3. Evaluation Loop
    top1_conv, top5_conv = 0, 0
    top1_dino, top5_dino = 0, 0
    top1_ens, top5_ens = 0, 0
    top1_probe, top5_probe = 0, 0
    total = 0

    start_eval_time = time.time()
    with torch.no_grad():
        for images, labels in eval_loader:
            images = images.to(device, non_blocking=True).float().div_(255.0)
            images = (images - mean) / std
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                logits_conv, logits_dino, logits_ens, _, _, emb_joint = student(images)
                logits_probe = joint_probe(emb_joint)

            # Conv Top-1 & Top-5
            _, pred5_conv = logits_conv.topk(5, dim=-1, largest=True, sorted=True)
            corr5_conv = pred5_conv.eq(labels.view(-1, 1).expand_as(pred5_conv))
            top1_conv += corr5_conv[:, 0].sum().item()
            top5_conv += corr5_conv[:, :5].sum().item()

            # DINO Top-1 & Top-5
            _, pred5_dino = logits_dino.topk(5, dim=-1, largest=True, sorted=True)
            corr5_dino = pred5_dino.eq(labels.view(-1, 1).expand_as(pred5_dino))
            top1_dino += corr5_dino[:, 0].sum().item()
            top5_dino += corr5_dino[:, :5].sum().item()

            # Ensemble Top-1 & Top-5
            _, pred5_ens = logits_ens.topk(5, dim=-1, largest=True, sorted=True)
            corr5_ens = pred5_ens.eq(labels.view(-1, 1).expand_as(pred5_ens))
            top1_ens += corr5_ens[:, 0].sum().item()
            top5_ens += corr5_ens[:, :5].sum().item()

            # Joint Probe Top-1 & Top-5
            _, pred5_probe = logits_probe.topk(5, dim=-1, largest=True, sorted=True)
            corr5_probe = pred5_probe.eq(labels.view(-1, 1).expand_as(pred5_probe))
            top1_probe += corr5_probe[:, 0].sum().item()
            top5_probe += corr5_probe[:, :5].sum().item()

            total += labels.size(0)

    eval_duration = time.time() - start_eval_time
    latency = (eval_duration / total) * 1000
    fps = total / eval_duration

    print("\n" + "=" * 75)
    print(f" FINAL [{split_name.upper()}] BENCHMARK RESULTS (OPTION B | NATIVE 64x64)")
    print("=" * 75)
    print(f"  ConvNeXt Stream Top-1  : {top1_conv/total*100:.2f}%  | Top-5: {top5_conv/total*100:.2f}%")
    print(f"  DINOv2 Stream Top-1    : {top1_dino/total*100:.2f}%  | Top-5: {top5_dino/total*100:.2f}%")
    print(f"  Simple Ensemble Top-1  : {top1_ens/total*100:.2f}%  | Top-5: {top5_ens/total*100:.2f}%")
    print(f"  Late Joint Probe Top-1 : {top1_probe/total*100:.2f}%  | Top-5: {top5_probe/total*100:.2f}%")
    print(f"  Inference Latency      : {latency:.2f} ms / image ({fps:.1f} FPS on {device})")
    print("=" * 75 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Option B Student")
    parser.add_argument("checkpoint", nargs="?", default="student_dual_option_b.pt", help="Path to checkpoint")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    args = parser.parse_args()

    evaluate_split("validation", checkpoint_path=args.checkpoint, batch_size=args.batch_size)
    evaluate_split("test", checkpoint_path=args.checkpoint, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
