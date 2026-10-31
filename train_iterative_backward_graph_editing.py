"""
Iterative Multi-Block Backward Cross-Backbone Graph Editing & Distillation
==========================================================================
Unified End-to-End Pipeline:
  Iteratively distills and physically prunes all 11 blocks of DINOv2 and corresponding
  ConvNeXt blocks backward in a single automated process (Block 11/35 -> Block 10/34 -> ... -> Block 0).

At each block k:
  1. Hooks intermediate representations at (ConvNeXt Block k, DINOv2 Block k).
  2. Constructs the Cross-Backbone Homophily Graph for the block.
  3. Prunes redundant filter nodes via Cumulative MFH.
  4. Physically slices Block k output rows AND Block k+1 input columns simultaneously.
  5. Cascades parameter collapse through both backbones while fine-tuning through downstream representation engine.
  6. Automatically checkpoints and advances backward to Block k-1!
"""

import os
import sys
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_from_disk
import timm

from dataloader import TinyImageNetHFDataset
from train_graph_editing import (
    GraphEditingGNN,
    LowRankPostAdapter,
    FastBatchLoader,
    DualBackboneGraphEditor,
    build_or_load_dual_backbone_graph_1792,
    run_pareto_two_stage,
    load_or_build_val_test_teacher_cache,
    evaluate_graph_editor,
)


def get_convnext_block_by_flat_index(model, flat_idx):
    """
    ConvNeXt-Base has 36 total blocks:
      Stage 0: 3 blocks  (0..2)   [128 channels]
      Stage 1: 3 blocks  (3..5)   [256 channels]
      Stage 2: 27 blocks (6..32)  [512 channels]
      Stage 3: 3 blocks  (33..35) [1024 channels]
    """
    curr = 0
    for s_idx, stage in enumerate(model.stages):
        num_b = len(stage.blocks)
        if curr + num_b > flat_idx:
            b_idx = flat_idx - curr
            return model.stages[s_idx].blocks[b_idx], s_idx, b_idx
        curr += num_b
    raise ValueError(f"Invalid flat ConvNeXt block index {flat_idx}")


class DualBackboneHookedPair(nn.Module):
    """
    Extracts intermediate representations at arbitrary pair of blocks:
      (ConvNeXt flat block idx, DINOv2 transformer block idx).
    """
    def __init__(self, conv_flat_idx: int, dino_block_idx: int):
        super().__init__()
        self.conv_flat_idx = conv_flat_idx
        self.dino_block_idx = dino_block_idx

        self.bb1 = timm.create_model("convnext_base", pretrained=True, num_classes=0)
        self.bb2 = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
        )
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

        self.inter_feats = {}
        conv_block, _, _ = get_convnext_block_by_flat_index(self.bb1, conv_flat_idx)
        dino_block = self.bb2.blocks[dino_block_idx]

        def hook_conv(module, inp, out):
            self.inter_feats["conv"] = out.mean([-2, -1])  # global pool -> [B, C]

        def hook_dino(module, inp, out):
            self.inter_feats["dino"] = out[:, 0]  # CLS token -> [B, D]

        self._h1 = conv_block.register_forward_hook(hook_conv)
        self._h2 = dino_block.register_forward_hook(hook_dino)

    @torch.no_grad()
    def forward(self, x):
        x_224 = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        f1_L = self.bb1(x_224)
        f2_L = self.bb2(x_224)
        f1_inter = self.inter_feats["conv"]
        f2_inter = self.inter_feats["dino"]
        return f1_inter, f2_inter, f1_L, f2_L


def extract_or_load_block_features(conv_flat_idx, dino_block_idx, cache_dir="cache_intermediate", device="cuda"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"inter_conv{conv_flat_idx}_dino{dino_block_idx}.pt")
    if os.path.exists(cache_path):
        print(f"[Cache] Loading cached features for ConvNeXt-{conv_flat_idx} & DINOv2-{dino_block_idx} from '{cache_path}'...")
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    print(f"\n[Cache] Extracting features for ConvNeXt Block {conv_flat_idx} & DINOv2 Block {dino_block_idx}...")
    transform_clean = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ds = load_from_disk("tiny-imagenet-200-clean/tiny-imagenet-200-clean")

    teachers = DualBackboneHookedPair(conv_flat_idx, dino_block_idx).to(device)
    teachers.eval()

    out = {}
    for split in ["train", "validation", "test"]:
        t0 = time.time()
        split_set = TinyImageNetHFDataset(ds[split], transform=transform_clean)
        loader = DataLoader(split_set, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

        f1_inter_list, f2_inter_list = [], []
        f1_L_list, f2_L_list = [], []
        y_list = []

        with torch.no_grad():
            for i, (images, labels) in enumerate(loader):
                images = images.to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    f1_inter, f2_inter, f1_L, f2_L = teachers(images)
                f1_inter_list.append(f1_inter.half().cpu())
                f2_inter_list.append(f2_inter.half().cpu())
                f1_L_list.append(f1_L.half().cpu())
                f2_L_list.append(f2_L.half().cpu())
                y_list.append(labels.cpu())

        out[split] = {
            "F1_inter": torch.cat(f1_inter_list, dim=0),
            "F2_inter": torch.cat(f2_inter_list, dim=0),
            "F1_L": torch.cat(f1_L_list, dim=0),
            "F2_L": torch.cat(f2_L_list, dim=0),
            "labels": torch.cat(y_list, dim=0),
        }
        print(f"  [{split.upper()}] Cached {len(split_set)} images in {time.time()-t0:.1f}s")

    torch.save(out, cache_path)
    del teachers
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def build_block_homophily_graph(F1_train, F2_train, top_k=32, device="cuda"):
    c_dim = F1_train.size(1)
    d_dim = F2_train.size(1)
    tot_dim = c_dim + d_dim

    n = min(25000, F1_train.size(0))
    X = torch.cat([F1_train[:n].float(), F2_train[:n].float()], dim=1).to(device)

    feat_mean = X.mean(dim=0, keepdim=True)
    feat_std = X.std(dim=0, keepdim=True).clamp(min=1e-4)
    Z = (X - feat_mean) / feat_std

    corr = (Z.t() @ Z) / float(n - 1)
    corr.fill_diagonal_(0.0)
    corr = corr.clamp(-0.99, 0.99)

    abs_corr = corr.abs()
    k_half = top_k // 2
    mask = torch.zeros_like(corr)

    _, idx_c_intra = torch.topk(abs_corr[:c_dim, :c_dim], k=min(k_half, c_dim - 1), dim=-1)
    _, idx_c_cross = torch.topk(abs_corr[:c_dim, c_dim:], k=min(k_half, d_dim), dim=-1)
    mask[:c_dim, :c_dim].scatter_(1, idx_c_intra, 1.0)
    mask[:c_dim, c_dim:].scatter_(1, idx_c_cross, 1.0)

    _, idx_d_intra = torch.topk(abs_corr[c_dim:, c_dim:], k=min(k_half, d_dim - 1), dim=-1)
    _, idx_d_cross = torch.topk(abs_corr[c_dim:, :c_dim], k=min(k_half, c_dim), dim=-1)
    mask[c_dim:, c_dim:].scatter_(1, idx_d_intra, 1.0)
    mask[c_dim:, :c_dim].scatter_(1, idx_d_cross, 1.0)

    adj_init = corr * mask
    node_mfh = (abs_corr * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    redundancy_order = torch.argsort(node_mfh, descending=True)

    return adj_init.cpu(), feat_mean.cpu(), feat_std.cpu(), redundancy_order.cpu()


class BlockStepGraphEditor(nn.Module):
    """
    Graph Editor for an intermediate block pair (ConvNeXt Block k & DINOv2 Block k).
    """
    def __init__(self, c_dim: int, d_dim: int, adj_init, feat_mean, feat_std, downstream_model=None, num_classes=200):
        super().__init__()
        self.c_dim = c_dim
        self.d_dim = d_dim
        self.tot_dim = c_dim + d_dim

        self.filter_bank_conv = nn.Linear(c_dim, c_dim)
        self.filter_bank_dino = nn.Linear(d_dim, d_dim)
        with torch.no_grad():
            nn.init.eye_(self.filter_bank_conv.weight)
            nn.init.zeros_(self.filter_bank_conv.bias)
            nn.init.eye_(self.filter_bank_dino.weight)
            nn.init.zeros_(self.filter_bank_dino.bias)

        self.register_buffer("conv_active_nodes", torch.arange(c_dim, dtype=torch.long))
        self.register_buffer("dino_active_nodes", torch.arange(d_dim, dtype=torch.long))
        self.register_buffer("alive_mask", torch.ones(self.tot_dim, dtype=torch.float32))

        self.gnn = GraphEditingGNN(
            num_nodes=self.tot_dim,
            num_layers=3,
            adj_init=adj_init,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )

        # Cascading transitions into downstream block
        down_c_in = downstream_model.filter_bank_conv.in_features if (downstream_model is not None and hasattr(downstream_model, "filter_bank_conv")) else c_dim
        down_d_in = downstream_model.filter_bank_dino.in_features if (downstream_model is not None and hasattr(downstream_model, "filter_bank_dino")) else d_dim

        self.trans_conv = nn.Linear(c_dim, down_c_in)
        self.trans_dino = nn.Linear(d_dim, down_d_in)
        with torch.no_grad():
            if c_dim == down_c_in:
                nn.init.eye_(self.trans_conv.weight)
            else:
                nn.init.orthogonal_(self.trans_conv.weight)
            nn.init.zeros_(self.trans_conv.bias)

            if d_dim == down_d_in:
                nn.init.eye_(self.trans_dino.weight)
            else:
                nn.init.orthogonal_(self.trans_dino.weight)
            nn.init.zeros_(self.trans_dino.bias)

        self.downstream_model = downstream_model
        if self.downstream_model is not None:
            self.downstream_model.eval()
            for p in self.downstream_model.parameters():
                p.requires_grad = False

        self.head_conv = nn.Linear(c_dim, num_classes)
        self.head_dino = nn.Linear(d_dim, num_classes)

    def count_total_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def physically_trim_conv(self):
        device = self.alive_mask.device
        dtype = self.filter_bank_conv.weight.dtype

        c_alive = self.alive_mask[self.conv_active_nodes] > 0.5
        if int(c_alive.sum().item()) < self.conv_active_nodes.numel():
            keep_c = torch.nonzero(c_alive, as_tuple=True)[0]
            new_c_nodes = self.conv_active_nodes[keep_c].contiguous()
            n_c = new_c_nodes.numel()

            new_fc_c = nn.Linear(self.c_dim, n_c).to(device=device, dtype=dtype)
            new_fc_c.weight.data.copy_(self.filter_bank_conv.weight.data[keep_c])
            new_fc_c.bias.data.copy_(self.filter_bank_conv.bias.data[keep_c])
            self.filter_bank_conv = new_fc_c
            self.conv_active_nodes = new_c_nodes

            # Cascades into downstream block input columns!
            out_c = self.trans_conv.weight.shape[0]
            new_trans_c = nn.Linear(n_c, out_c).to(device=device, dtype=dtype)
            new_trans_c.weight.data.copy_(self.trans_conv.weight.data[:, keep_c])
            new_trans_c.bias.data.copy_(self.trans_conv.bias.data)
            self.trans_conv = new_trans_c

        self.gnn.trim_dead_hop1_edges(self.alive_mask)

    @torch.no_grad()
    def physically_trim_dino(self):
        device = self.alive_mask.device
        dtype = self.filter_bank_dino.weight.dtype

        d_alive = self.alive_mask[self.c_dim + self.dino_active_nodes] > 0.5
        if int(d_alive.sum().item()) < self.dino_active_nodes.numel():
            keep_d = torch.nonzero(d_alive, as_tuple=True)[0]
            new_d_nodes = self.dino_active_nodes[keep_d].contiguous()
            n_d = new_d_nodes.numel()

            new_fc_d = nn.Linear(self.d_dim, n_d).to(device=device, dtype=dtype)
            new_fc_d.weight.data.copy_(self.filter_bank_dino.weight.data[keep_d])
            new_fc_d.bias.data.copy_(self.filter_bank_dino.bias.data[keep_d])
            self.filter_bank_dino = new_fc_d
            self.dino_active_nodes = new_d_nodes

            # Cascades into downstream block input columns!
            out_d = self.trans_dino.weight.shape[0]
            new_trans_d = nn.Linear(n_d, out_d).to(device=device, dtype=dtype)
            new_trans_d.weight.data.copy_(self.trans_dino.weight.data[:, keep_d])
            new_trans_d.bias.data.copy_(self.trans_dino.bias.data)
            self.trans_dino = new_trans_d

    @torch.no_grad()
    def physically_trim_block(self):
        self.physically_trim_dino()
        self.physically_trim_conv()

    def forward(self, f1_in, f2_in):
        B = f1_in.size(0)

        c_out = self.filter_bank_conv(f1_in)
        d_out = self.filter_bank_dino(f2_in)

        X_raw = torch.zeros(B, self.tot_dim, device=f1_in.device, dtype=f1_in.dtype)
        X_raw.index_copy_(1, self.conv_active_nodes, c_out)
        X_raw.index_copy_(1, self.c_dim + self.dino_active_nodes, d_out)
        X_raw = X_raw * self.alive_mask.view(1, -1)

        X_gnn = self.gnn(X_raw, self.alive_mask)

        c_in_down = X_gnn[:, self.conv_active_nodes]
        d_in_down = X_gnn[:, self.c_dim + self.dino_active_nodes]

        f1_pred = self.trans_conv(c_in_down)
        f2_pred = self.trans_dino(d_in_down)

        if self.downstream_model is not None:
            res = self.downstream_model(f1_pred, f2_pred)
            logits_conv, logits_dino, logits_ens = res[0], res[1], res[2]
        else:
            logits_conv = self.head_conv(f1_pred)
            logits_dino = self.head_dino(f2_pred)
            logits_ens = 0.5 * logits_conv + 0.5 * logits_dino

        return logits_conv, logits_dino, logits_ens, f1_pred, f2_pred


@torch.no_grad()
def evaluate_block_editor(model, eval_dict, device, batch_size=256):
    model.eval()
    F1_in = eval_dict["F1_inter"]
    F2_in = eval_dict["F2_inter"]
    labels = eval_dict["labels"]
    n = labels.size(0)

    top1_c, top1_d, top1_e = 0, 0, 0
    top5_e = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        f1_b = F1_in[start:end].to(device, non_blocking=True).float()
        f2_b = F2_in[start:end].to(device, non_blocking=True).float()
        y_b = labels[start:end].to(device, non_blocking=True)

        logits_c, logits_d, logits_e, _, _ = model(f1_b, f2_b)

        top1_c += (logits_c.argmax(dim=-1) == y_b).sum().item()
        top1_d += (logits_d.argmax(dim=-1) == y_b).sum().item()
        top1_e += (logits_e.argmax(dim=-1) == y_b).sum().item()

        _, pred5 = logits_e.topk(5, dim=-1, largest=True, sorted=True)
        top5_e += pred5.eq(y_b.view(-1, 1).expand_as(pred5)).any(dim=-1).sum().item()

    return {
        "conv_top1": 100.0 * top1_c / n,
        "dino_top1": 100.0 * top1_d / n,
        "ens_top1": 100.0 * top1_e / n,
        "ens_top5": 100.0 * top5_e / n,
    }


def train_single_backward_block(
    conv_flat_idx: int,
    dino_block_idx: int,
    downstream_model,
    cache_data,
    args,
    device="cuda",
):
    print(f"\n{'='*80}")
    print(f" DISTILLING & PRUNING BLOCK PAIR: ConvNeXt-{conv_flat_idx} & DINOv2-{dino_block_idx}")
    print(f"{'='*80}")

    train_f1 = cache_data["train"]["F1_inter"]
    train_f2 = cache_data["train"]["F2_inter"]
    train_f1_target = cache_data["train"]["F1_L"]
    train_f2_target = cache_data["train"]["F2_L"]
    y_train = cache_data["train"]["labels"]

    c_dim = train_f1.size(1)
    d_dim = train_f2.size(1)

    adj_init, feat_mean, feat_std, redundancy_order = build_block_homophily_graph(
        train_f1, train_f2, top_k=args.top_k, device=device
    )

    model = BlockStepGraphEditor(
        c_dim=c_dim,
        d_dim=d_dim,
        adj_init=adj_init,
        feat_mean=feat_mean,
        feat_std=feat_std,
        downstream_model=downstream_model,
        num_classes=200,
    ).to(device)

    # Pareto Deletion Setup
    c_redundant = redundancy_order[redundancy_order < c_dim]
    d_redundant = redundancy_order[redundancy_order >= c_dim]

    max_dino_del = 1.0 - args.dino_keep  # e.g. 0.55
    max_conv_del = 1.0 - args.conv_keep  # e.g. 0.80

    p1_epochs = getattr(args, "p1_epochs", 30)
    p2_epochs = getattr(args, "p2_epochs", 30)
    p3_epochs = getattr(args, "p3_epochs", 10)

    pre_eval = evaluate_block_editor(model, cache_data["validation"], device)
    print(f"  Step 0 Pre-Trim Val: Conv={pre_eval['conv_top1']:.2f}% | DINO={pre_eval['dino_top1']:.2f}% | Ens={pre_eval['ens_top1']:.2f}%")

    n_samples = train_f1.size(0)

    def run_one_epoch(opt):
        model.train()
        perm = torch.randperm(n_samples)
        total_loss, total_bs = 0.0, 0
        for start in range(0, n_samples, args.batch_size):
            end = min(start + args.batch_size, n_samples)
            idx = perm[start:end]
            f1_b = train_f1[idx].to(device, non_blocking=True).float()
            f2_b = train_f2[idx].to(device, non_blocking=True).float()
            f1_t = train_f1_target[idx].to(device, non_blocking=True).float()
            f2_t = train_f2_target[idx].to(device, non_blocking=True).float()
            y_b = y_train[idx].to(device, non_blocking=True)

            logits_c, logits_d, logits_e, f1_p, f2_p = model(f1_b, f2_b)

            if f1_p.shape == f1_t.shape and f2_p.shape == f2_t.shape:
                loss_match = F.smooth_l1_loss(f1_p, f1_t) + F.smooth_l1_loss(f2_p, f2_t)
            else:
                loss_match = 0.0

            kl_c = F.kl_div(F.log_softmax(logits_c / 2.0, dim=-1), F.softmax(logits_d.detach() / 2.0, dim=-1), reduction="batchmean") * 4.0
            kl_d = F.kl_div(F.log_softmax(logits_d / 2.0, dim=-1), F.softmax(logits_c.detach() / 2.0, dim=-1), reduction="batchmean") * 4.0
            loss_ce = 0.5 * F.cross_entropy(logits_c, y_b) + 0.5 * F.cross_entropy(logits_d, y_b)
            loss = loss_ce + 0.3 * (kl_c + kl_d) + (3.0 * loss_match if isinstance(loss_match, torch.Tensor) else 0.0)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = y_b.size(0)
            total_loss += loss.item() * bs
            total_bs += bs
        return total_loss / total_bs

    # =========================================================================
    # PHASE 1: DINOv2 PARETO SWEEP (ConvNeXt is 100% Intact Anchor)
    # =========================================================================
    print(f"\n  [PHASE 1] DINOv2 Pareto Sweep (0% -> {max_dino_del*100:.0f}% deleted over {p1_epochs} epochs | ConvNeXt is Intact Anchor)...")
    opt1 = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=p1_epochs, eta_min=1e-5)

    best_dino_score = -1.0
    best_dino_state = copy.deepcopy(model.state_dict())
    t0 = time.time()

    for ep in range(1, p1_epochs + 1):
        d_del_ratio = 0.0 if ep < 2 else max_dino_del * ((ep - 1) / float(p1_epochs - 1))
        target_del_d = int(round(d_del_ratio * len(d_redundant)))

        # Conv is 100% alive anchor, DINO gets progressively deleted
        model.alive_mask.fill_(1.0)
        for idx in d_redundant[:target_del_d].tolist():
            model.alive_mask[idx] = 0.0

        loss = run_one_epoch(opt1)
        sched1.step()

        val_m = evaluate_block_editor(model, cache_data["validation"], device)
        score = 0.5 * (val_m["ens_top1"] + val_m["dino_top1"])
        is_peak = False
        if target_del_d >= int(0.7 * max_dino_del * len(d_redundant)) and score > best_dino_score:
            best_dino_score = score
            best_dino_state = copy.deepcopy(model.state_dict())
            is_peak = True

        if ep % 5 == 0 or ep == p1_epochs or is_peak:
            peak_str = " << NEW DINO PEAK!" if is_peak else ""
            d_alive = d_dim - target_del_d
            print(f"    [P1 Ep {ep:2d}/{p1_epochs}] DINO Deleted: {100.0*target_del_d/d_dim:4.1f}% (Alive: DINO={d_alive}/{d_dim}, Conv={c_dim}/{c_dim}) | Val: [Conv: {val_m['conv_top1']:.2f}% | DINO: {val_m['dino_top1']:.2f}% | Ens: {val_m['ens_top1']:.2f}%]{peak_str} | {time.time()-t0:.1f}s")

    # Lock & physically trim Phase 1 DINOv2 subgraph
    model.load_state_dict(best_dino_state)
    model.physically_trim_dino()
    print(f"  >>> LOCKED & PHYSICALLY TRIMMED DINOv2 SUBGRAPH: {model.dino_active_nodes.numel()}/{d_dim} Alive Filters!")

    # =========================================================================
    # PHASE 2: CONVNEXT PARETO SWEEP (DINOv2 is Locked Anchor)
    # =========================================================================
    print(f"\n  [PHASE 2] ConvNeXt Pareto Sweep (0% -> {max_conv_del*100:.0f}% deleted over {p2_epochs} epochs | DINOv2 is Locked Anchor)...")
    opt2 = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.8, weight_decay=1e-4)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=p2_epochs, eta_min=1e-5)

    best_p2_score = -1.0
    best_p2_state = copy.deepcopy(model.state_dict())
    t0 = time.time()

    for ep in range(1, p2_epochs + 1):
        c_del_ratio = 0.0 if ep < 2 else max_conv_del * ((ep - 1) / float(p2_epochs - 1))
        target_del_c = int(round(c_del_ratio * len(c_redundant)))

        # Keep DINO locked, prune Conv
        model.alive_mask[:c_dim].fill_(1.0)
        for idx in c_redundant[:target_del_c].tolist():
            model.alive_mask[idx] = 0.0

        loss = run_one_epoch(opt2)
        sched2.step()

        val_m = evaluate_block_editor(model, cache_data["validation"], device)
        score = 0.5 * (val_m["ens_top1"] + val_m["conv_top1"])
        is_peak = False
        if target_del_c >= int(0.7 * max_conv_del * len(c_redundant)) and score > best_p2_score:
            best_p2_score = score
            best_p2_state = copy.deepcopy(model.state_dict())
            is_peak = True

        if ep % 5 == 0 or ep == p2_epochs or is_peak:
            peak_str = " << NEW JOINT PARETO PEAK!" if is_peak else ""
            c_alive = c_dim - target_del_c
            print(f"    [P2 Ep {ep:2d}/{p2_epochs}] Conv Deleted: {100.0*target_del_c/c_dim:4.1f}% (Alive: Conv={c_alive}/{c_dim}, DINO={model.dino_active_nodes.numel()}/{d_dim}) | Val: [Conv: {val_m['conv_top1']:.2f}% | DINO: {val_m['dino_top1']:.2f}% | Ens: {val_m['ens_top1']:.2f}%]{peak_str} | {time.time()-t0:.1f}s")

    # Lock & physically trim Phase 2 ConvNeXt subgraph
    model.load_state_dict(best_p2_state)
    model.physically_trim_conv()
    print(f"  >>> LOCKED & PHYSICALLY TRIMMED CONVNEXT SUBGRAPH: {model.conv_active_nodes.numel()}/{c_dim} Alive Filters!")

    # =========================================================================
    # PHASE 3: FINE-TUNING PHYSICALLY TRIMMED SUBGRAPH
    # =========================================================================
    print(f"\n  [PHASE 3] Fine-Tuning Physically Trimmed Subgraph for {p3_epochs} Epochs...")
    opt3 = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.35, weight_decay=1e-4)
    sched3 = torch.optim.lr_scheduler.CosineAnnealingLR(opt3, T_max=p3_epochs, eta_min=1e-5)

    best_final_val = -1.0
    best_final_model = None

    for ep in range(1, p3_epochs + 1):
        loss = run_one_epoch(opt3)
        sched3.step()

        val_m = evaluate_block_editor(model, cache_data["validation"], device)
        if val_m["ens_top1"] > best_final_val:
            best_final_val = val_m["ens_top1"]
            best_final_model = copy.deepcopy(model)

        if ep % 5 == 0 or ep == p3_epochs:
            print(f"    [P3 Ep {ep:2d}/{p3_epochs}] Trimmed Subgraph Val: [Conv: {val_m['conv_top1']:.2f}% | DINO: {val_m['dino_top1']:.2f}% | Ens: {val_m['ens_top1']:.2f}% (Top-5: {val_m['ens_top5']:.2f}%)]")

    eval_model = best_final_model if best_final_model is not None else model
    val_final = evaluate_block_editor(eval_model, cache_data["validation"], device)
    test_final = evaluate_block_editor(eval_model, cache_data["test"], device)

    c_saved = (c_dim - eval_model.conv_active_nodes.numel()) * c_dim
    d_saved = (d_dim - eval_model.dino_active_nodes.numel()) * d_dim
    total_saved = c_saved + d_saved

    print(f"\n  >>> COMPLETED PARETO BLOCK: ConvNeXt-{conv_flat_idx} ({eval_model.conv_active_nodes.numel()}/{c_dim}) & DINOv2-{dino_block_idx} ({eval_model.dino_active_nodes.numel()}/{d_dim})")
    print(f"      Params Dropped: -{total_saved/1e6:.3f}M | Val: Ens={val_final['ens_top1']:.2f}% | Test: Ens={test_final['ens_top1']:.2f}%")

    return eval_model, val_final, test_final, total_saved


def get_or_train_step0_model(ckpt_path="graph_edited_dual_backbone.pt", device="cuda"):
    """
    Loads the Step 0 Representation Engine (Block 35 & 11) if it already exists,
    or automatically trains and saves it on the fly if not found!
    """
    if os.path.exists(ckpt_path):
        print(f"[Step 0] Found existing Step 0 representation checkpoint: '{ckpt_path}'")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model" in ckpt:
            model = ckpt["model"]
        else:
            train_data = torch.load("tiny_dual_targets.pt", map_location="cpu", weights_only=False)
            adj_init, feat_mean, feat_std, _ = build_or_load_dual_backbone_graph_1792(
                train_data["F1"], train_data["F2"], device=device
            )
            model = DualBackboneGraphEditor(
                adj_init=adj_init,
                feat_mean=feat_mean,
                feat_std=feat_std,
                num_classes=200,
            ).to(device)
            if "conv_active_nodes" in ckpt:
                model.conv_active_nodes = ckpt["conv_active_nodes"].to(device)
                model.dino_active_nodes = ckpt["dino_active_nodes"].to(device)
                model.alive_mask = ckpt["alive_mask"].to(device)
                model.physically_trim_deleted_filters()
            model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    print(f"\n{'='*84}\n STEP 0: TRAINING INITIAL DUAL-BACKBONE REPRESENTATION ENGINE (Block 35 & 11)\n{'='*84}")
    train_data = torch.load("tiny_dual_targets.pt", map_location="cpu", weights_only=False)
    val_test_data = load_or_build_val_test_teacher_cache(device=device)

    F1_train = train_data["F1"]
    F2_train = train_data["F2"]
    t1_logits_train = train_data["t1_logits"]
    t2_logits_train = train_data["t2_logits"]
    img_cache = torch.load("tiny_images_uint8.pt", map_location="cpu", weights_only=False)
    y_train = img_cache["train"]["labels"]

    joint_probe = nn.Linear(1792, 200).to(device)
    joint_probe.load_state_dict(train_data["joint_probe_state"])
    joint_probe.eval()
    for p in joint_probe.parameters():
        p.requires_grad = False

    adj_init, feat_mean, feat_std, redundancy_order = build_or_load_dual_backbone_graph_1792(
        F1_train=F1_train,
        F2_train=F2_train,
        top_k=32,
        device=device,
    )

    model = DualBackboneGraphEditor(
        adj_init=adj_init,
        feat_mean=feat_mean,
        feat_std=feat_std,
        probe_conv_state=train_data.get("probe_conv_state"),
        probe_dino_state=train_data.get("probe_dino_state"),
        num_classes=200,
    ).to(device)

    train_loader = FastBatchLoader(
        F1_train, F2_train, y_train, t1_logits_train, t2_logits_train,
        batch_size=256, shuffle=True
    )

    class DummyArgs:
        min_delete = 0.50
        temp = 2.0
        save_ckpt = ckpt_path

    model = run_pareto_two_stage(
        model=model,
        train_loader=train_loader,
        val_test_data=val_test_data,
        joint_probe=joint_probe,
        redundancy_order=redundancy_order,
        args=DummyArgs(),
        device=device,
    )

    torch.save({"model": model}, ckpt_path)
    print(f"[Step 0] Saved trimmed Step 0 model to '{ckpt_path}'!\n")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def main():
    parser = argparse.ArgumentParser(description="Iterative Multi-Block Backward Cross-Backbone Graph Editing")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs per backward block (default: 15)")
    parser.add_argument("--p1_epochs", type=int, default=30, help="Phase 1 DINO Pareto sweep epochs (default: 30)")
    parser.add_argument("--p2_epochs", type=int, default=30, help="Phase 2 ConvNeXt Pareto sweep epochs (default: 30)")
    parser.add_argument("--p3_epochs", type=int, default=10, help="Phase 3 fine-tuning epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--conv_keep", type=float, default=0.20, help="ConvNeXt alive filter ratio (default: 0.20 = 80% deleted)")
    parser.add_argument("--dino_keep", type=float, default=0.45, help="DINOv2 alive filter ratio (default: 0.45 = 55% deleted)")
    parser.add_argument("--num_blocks", type=int, default=11, help="Number of blocks to propagate backward (1 to 11)")
    parser.add_argument("--initial_ckpt", type=str, default="graph_edited_dual_backbone.pt", help="Initial Step 1 checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 84)
    print(" ITERATIVE MULTI-BLOCK BACKWARD GRAPH EDITING & DISTILLATION PIPELINE")
    print(f" Propagating backward across {args.num_blocks} blocks sequentially...")
    print(f" ConvNeXt Keep Ratio: {args.conv_keep*100:.0f}% (80% pruned) | DINOv2 Keep Ratio: {args.dino_keep*100:.0f}% (55% pruned)")
    print("=" * 84 + "\n")

    # Load or automatically train Step 0 representation model (Block 35 & Block 11)
    current_downstream = get_or_train_step0_model(args.initial_ckpt, device=device)

    # Block Schedule: ConvNeXt flat indices [34, 33, 32, 31, ...] paired with DINOv2 indices [10, 9, 8, 7, ...]
    conv_block_schedule = [35 - step for step in range(1, args.num_blocks + 1)]
    dino_block_schedule = [11 - step for step in range(1, args.num_blocks + 1)]

    history = []
    cumulative_params_dropped = 12.02e6  # ~12M already dropped in Step 1 & 2

    for step, (c_idx, d_idx) in enumerate(zip(conv_block_schedule, dino_block_schedule), start=1):
        print(f"\n{'#'*84}")
        print(f" ITERATION [{step}/{args.num_blocks}]: Propagating into ConvNeXt Block {c_idx} & DINOv2 Block {d_idx}")
        print(f"{'#'*84}")

        cache_data = extract_or_load_block_features(c_idx, d_idx, device=device)

        step_model, val_m, test_m, saved_params = train_single_backward_block(
            conv_flat_idx=c_idx,
            dino_block_idx=d_idx,
            downstream_model=current_downstream,
            cache_data=cache_data,
            args=args,
            device=device,
        )

        cumulative_params_dropped += saved_params
        current_downstream = step_model

        ckpt_path = f"graph_edited_iterative_block_c{c_idx}_d{d_idx}.pt"
        torch.save(
            {
                "step": step,
                "conv_idx": c_idx,
                "dino_idx": d_idx,
                "model": step_model,
                "val_metrics": val_m,
                "test_metrics": test_m,
                "cumulative_params_dropped": cumulative_params_dropped,
            },
            ckpt_path,
        )
        print(f"  [Saved Checkpoint] -> '{ckpt_path}'")

        history.append({
            "step": step,
            "c_idx": c_idx,
            "d_idx": d_idx,
            "val_ens": val_m["ens_top1"],
            "test_ens": test_m["ens_top1"],
            "saved_params": saved_params,
            "cum_saved": cumulative_params_dropped,
        })

    # Final Summary Table across all blocks
    print("\n" + "=" * 88)
    print(" COMPLETE ITERATIVE BACKWARD GRAPH EDITING BENCHMARK TABLE")
    print("=" * 88)
    print(f" {'Step':<5} | {'ConvNeXt Block':<15} | {'DINOv2 Block':<13} | {'Val Ens (%)':<12} | {'Test Ens (%)':<12} | {'Cumulative Pruned':<18}")
    print("-" * 88)
    for h in history:
        print(f" {h['step']:<5} | Block {h['c_idx']:<9} | Block {h['d_idx']:<7} | {h['val_ens']:<12.2f} | {h['test_ens']:<12.2f} | -{h['cum_saved']/1e6:<6.2f}M params")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()
