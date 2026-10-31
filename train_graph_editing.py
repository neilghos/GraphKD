import os
import copy
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_from_disk

from cache import DualBackbone
from dataloader import TinyImageNetHFDataset, FastBatchLoader


@torch.no_grad()
def load_or_build_val_test_teacher_cache(
    cache_path="tiny_val_test_teacher_feats.pt",
    data_path="tiny-imagenet-200-clean/tiny-imagenet-200-clean",
    batch_size=128,
    device="cuda",
):
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    print("=" * 78)
    print(f" EXTRACTING VAL & TEST FROZEN FEATURE BANKS -> {cache_path}")
    print("=" * 78)
    transform_clean = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = load_from_disk(data_path)
    teachers = DualBackbone().to(device)
    teachers.eval()

    out = {}
    for split in ["validation", "test"]:
        split_set = TinyImageNetHFDataset(ds[split], transform=transform_clean)
        loader = DataLoader(split_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        f1_list, f2_list, y_list = [], [], []
        t0 = time.time()
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                f1, f2 = teachers(images)
            f1_list.append(f1.half().cpu())
            f2_list.append(f2.half().cpu())
            y_list.append(labels.cpu())
        out[split] = {
            "F1": torch.cat(f1_list, dim=0),
            "F2": torch.cat(f2_list, dim=0),
            "labels": torch.cat(y_list, dim=0),
        }
        print(f"  Cached '{split}' ({len(split_set)} images) in {time.time() - t0:.1f}s")

    torch.save(out, cache_path)
    del teachers
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def build_or_load_dual_backbone_graph_1792(
    F1_train: torch.Tensor,
    F2_train: torch.Tensor,
    cache_path="graph_editing_adj_1792.pt",
    top_k=32,
    max_samples=25000,
    device="cuda",
):
    if os.path.exists(cache_path):
        print(f"[GraphEdit] Loading cached 1792x1792 filter graph from '{cache_path}'...")
        d = torch.load(cache_path, map_location="cpu", weights_only=True)
        return d["adj_init"], d["feat_mean"], d["feat_std"], d["node_redundancy_order"]

    print(f"[GraphEdit] Computing 1792x1792 Cross-Backbone Filter Graph (top_k={top_k})...")
    n = min(max_samples, F1_train.size(0))
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

    _, idx_c_intra = torch.topk(abs_corr[:1024, :1024], k=k_half, dim=-1)
    _, idx_c_cross = torch.topk(abs_corr[:1024, 1024:], k=k_half, dim=-1)
    mask[:1024, :1024].scatter_(1, idx_c_intra, 1.0)
    mask[:1024, 1024:].scatter_(1, idx_c_cross, 1.0)

    _, idx_d_intra = torch.topk(abs_corr[1024:, 1024:], k=k_half, dim=-1)
    _, idx_d_cross = torch.topk(abs_corr[1024:, :1024], k=k_half, dim=-1)
    mask[1024:, 1024:].scatter_(1, idx_d_intra, 1.0)
    mask[1024:, :1024].scatter_(1, idx_d_cross, 1.0)

    mask = torch.maximum(mask, mask.t())
    adj_init = (corr * mask).cpu()

    redundancy_score = (abs_corr * mask).sum(dim=-1).cpu()
    node_redundancy_order = torch.argsort(redundancy_score, descending=True)

    mean_deg = (adj_init.abs() > 1e-7).float().sum(dim=-1).mean().item()
    print(f"[GraphEdit] Built 1792x1792 graph | Avg degree: {mean_deg:.1f} (50% Intra / 50% Cross-Backbone)")

    feat_mean_cpu = feat_mean.cpu()
    feat_std_cpu = feat_std.cpu()
    torch.save(
        {
            "adj_init": adj_init,
            "feat_mean": feat_mean_cpu,
            "feat_std": feat_std_cpu,
            "node_redundancy_order": node_redundancy_order,
        },
        cache_path,
    )
    return adj_init, feat_mean_cpu, feat_std_cpu, node_redundancy_order


class LowRankPostAdapter(nn.Module):
    """
    Compact Residual Low-Rank Post-GNN Adapter (0.06M params instead of 1.05M dense matrix!).
    Initialized to exact Identity (up.weight = 0) so Step-0 is 100% identical to the teacher!
    """
    def __init__(self, dim: int, rank: int = 32):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, dim))
        self.bias = nn.Parameter(torch.zeros(1, dim))
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        with torch.no_grad():
            nn.init.orthogonal_(self.down.weight)
            nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.scale * x + self.up(self.down(x)) + self.bias


class GraphEditingGNN(nn.Module):
    """
    Sparse-Edge 3-Layer 1792-Node Filter Graph Editing Engine.
    - Stores ONLY active graph edges (`edge_dst`, `edge_src`, `edge_weights`: ~83K edges/layer = 0.25M params)
      instead of 12.85M dense 1792x1792 matrices!
    - When `trim_dead_hop1_edges(alive_mask)` is called, any Hop-1 edge originating from a deleted node
      is physically removed from `edge_dst_0, edge_src_0, edge_weights[0]`, shrinking the GNN further!
    """
    def __init__(self, num_nodes=1792, num_layers=3, adj_init=None, feat_mean=None, feat_std=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers

        if adj_init is None:
            adj_init = torch.zeros(num_nodes, num_nodes)
        if feat_mean is None:
            feat_mean = torch.zeros(1, num_nodes)
        if feat_std is None:
            feat_std = torch.ones(1, num_nodes)

        edge_mask = (adj_init.abs() > 1e-7)
        edge_mask.fill_diagonal_(False)
        dst_idx, src_idx = torch.nonzero(edge_mask, as_tuple=True)

        self.register_buffer("feat_mean", feat_mean.view(num_nodes, 1).clone())
        self.register_buffer("feat_std", feat_std.view(num_nodes, 1).clamp(min=1e-4).clone())

        # Per-layer sparse edge lists so Hop 1 can physically trim outgoing edges from deleted nodes!
        for l in range(num_layers):
            self.register_buffer(f"edge_dst_{l}", dst_idx.long().clone())
            self.register_buffer(f"edge_src_{l}", src_idx.long().clone())

        self.edge_weights = nn.ParameterList([
            nn.Parameter(adj_init[dst_idx, src_idx].clone()) for _ in range(num_layers)
        ])

        self.alive_scales = nn.ParameterList([
            nn.Parameter(torch.full((num_nodes, 1), 0.05)) for _ in range(num_layers)
        ])
        self.deleted_scales = nn.ParameterList([
            nn.Parameter(torch.full((num_nodes, 1), 1.0)) for _ in range(num_layers)
        ])

        self.hop_mix = nn.Parameter(torch.tensor([0.50, 0.30, 0.20]).view(1, 3, 1).repeat(num_nodes, 1, 1))

    @torch.no_grad()
    def trim_dead_hop1_edges(self, alive_mask: torch.Tensor):
        """
        Physically removes Hop-1 edges whose source node has been deleted (`alive_mask[src] == 0`),
        literally trimming the parameter tensor `self.edge_weights[0]` as nodes are deleted!
        """
        src0 = self.edge_src_0
        dst0 = self.edge_dst_0
        keep_edge = alive_mask[src0] > 0.5
        if int(keep_edge.sum().item()) < src0.numel():
            self.edge_src_0 = src0[keep_edge].contiguous()
            self.edge_dst_0 = dst0[keep_edge].contiguous()
            self.edge_weights[0] = nn.Parameter(self.edge_weights[0].data[keep_edge].contiguous())

    def _get_layer_adj(self, layer_idx, source_mask=None):
        dst = getattr(self, f"edge_dst_{layer_idx}")
        src = getattr(self, f"edge_src_{layer_idx}")
        w_val = torch.tanh(self.edge_weights[layer_idx].float())
        if source_mask is not None:
            w_val = w_val * source_mask[src]

        W = torch.zeros(self.num_nodes, self.num_nodes, device=dst.device, dtype=torch.float32)
        W.index_put_((dst, src), w_val)
        deg = W.abs().sum(dim=-1, keepdim=True).clamp(min=1e-5)
        return W / deg

    def forward(self, X_raw, alive_mask):
        with torch.amp.autocast("cuda", enabled=False):
            X_f32 = X_raw.float()
            alive_col = alive_mask.view(self.num_nodes, 1).float()
            deleted_col = 1.0 - alive_col

            H = X_f32.t()  # [1792, B]
            H_z = ((H - self.feat_mean) / self.feat_std).clamp(-12.0, 12.0)
            H_0 = H_z * alive_col

            A1 = self._get_layer_adj(0, source_mask=alive_col.squeeze(-1))
            m1 = torch.matmul(A1, H_0)
            H_1 = alive_col * (H_0 + 0.5 * torch.tanh(self.alive_scales[0]) * F.gelu(m1)) + \
                  deleted_col * (self.deleted_scales[0] * m1)

            A2 = self._get_layer_adj(1, source_mask=None)
            m2 = torch.matmul(A2, F.gelu(H_1))
            H_2 = alive_col * (H_1 + 0.5 * torch.tanh(self.alive_scales[1]) * m2) + \
                  deleted_col * (H_1 + self.deleted_scales[1] * m2)

            A3 = self._get_layer_adj(2, source_mask=None)
            m3 = torch.matmul(A3, F.gelu(H_2))
            H_3 = alive_col * (H_2 + 0.5 * torch.tanh(self.alive_scales[2]) * m3) + \
                  deleted_col * (H_2 + self.deleted_scales[2] * m3)

            hops_stack = torch.stack([H_1, H_2, H_3], dim=1)  # [1792, 3, B]
            w_hops = F.softmax(self.hop_mix, dim=1)           # [1792, 3, 1]
            H_recon = (hops_stack * w_hops).sum(dim=1)        # [1792, B]

            H_final_z = (alive_col * H_3 + deleted_col * H_recon).clamp(-12.0, 12.0)
            H_out = H_final_z * self.feat_std + self.feat_mean

            return H_out.t().to(X_raw.dtype)


class DualBackboneGraphEditor(nn.Module):
    """
    Direct Dual-Backbone Filter Bank + Sparse 1792-Node Graph Editing Engine with Physical Filter Trimming.

    When `physically_trim_deleted_filters()` is called:
      - `filter_bank_conv` is physically sliced from `[1024, 1024]` -> `[K_conv, 1024]` (e.g. `[275, 1024]`)!
      - `filter_bank_dino` is physically sliced from `[768, 768]`   -> `[K_dino, 768]`  (e.g. `[237, 768]`)!
      - `gnn.edge_weights[0]` physically trims all outgoing Hop-1 edges from deleted nodes!
      - Total `.numel()` parameter count drops from 2.39M (Pre-Trim) -> ~1.04M (Post-Trim)!
    """
    def __init__(
        self,
        adj_init=None,
        feat_mean=None,
        feat_std=None,
        probe_conv_state=None,
        probe_dino_state=None,
        num_classes=200,
    ):
        super().__init__()
        self.num_nodes = 1792

        self.filter_bank_conv = nn.Linear(1024, 1024)
        self.filter_bank_dino = nn.Linear(768, 768)
        with torch.no_grad():
            nn.init.eye_(self.filter_bank_conv.weight)
            nn.init.zeros_(self.filter_bank_conv.bias)
            nn.init.eye_(self.filter_bank_dino.weight)
            nn.init.zeros_(self.filter_bank_dino.bias)

        # Active node index maps (initially all 1024 Conv nodes and all 768 DINO nodes)
        self.register_buffer("conv_active_nodes", torch.arange(1024, dtype=torch.long))
        self.register_buffer("dino_active_nodes", torch.arange(768, dtype=torch.long))
        self.register_buffer("alive_mask", torch.ones(1792, dtype=torch.float32))

        self.gnn = GraphEditingGNN(
            num_nodes=1792,
            num_layers=3,
            adj_init=adj_init,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )

        # Compact Low-Rank Post-GNN Adapters (0.11M total instead of 1.64M dense matrices!)
        self.post_conv = LowRankPostAdapter(1024, rank=32)
        self.post_dino = LowRankPostAdapter(768, rank=32)

        self.head_conv = nn.Linear(1024, num_classes)
        self.head_dino = nn.Linear(768, num_classes)
        if probe_conv_state is not None:
            self.head_conv.load_state_dict(probe_conv_state)
        if probe_dino_state is not None:
            self.head_dino.load_state_dict(probe_dino_state)

    def count_total_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def physically_trim_deleted_filters(self):
        """
        Physically slices deleted filter rows out of `self.filter_bank_conv` and `self.filter_bank_dino`,
        and physically trims dead Hop-1 outgoing edges from `self.gnn`!
        Can be called repeatedly as filters are progressively deleted!
        """
        device = self.alive_mask.device
        dtype = self.filter_bank_conv.weight.dtype

        # 1. Trim ConvNeXt Filter Bank to currently alive rows
        c_global_alive = self.alive_mask[self.conv_active_nodes] > 0.5
        if int(c_global_alive.sum().item()) < self.conv_active_nodes.numel():
            keep_c = torch.nonzero(c_global_alive, as_tuple=True)[0]
            new_c_nodes = self.conv_active_nodes[keep_c].contiguous()
            n_c = new_c_nodes.numel()
            new_fc_c = nn.Linear(1024, n_c).to(device=device, dtype=dtype)
            new_fc_c.weight.data.copy_(self.filter_bank_conv.weight.data[keep_c])
            new_fc_c.bias.data.copy_(self.filter_bank_conv.bias.data[keep_c])
            self.filter_bank_conv = new_fc_c
            self.conv_active_nodes = new_c_nodes

        # 2. Trim DINOv2 Filter Bank to currently alive rows
        d_global_alive = self.alive_mask[1024 + self.dino_active_nodes] > 0.5
        if int(d_global_alive.sum().item()) < self.dino_active_nodes.numel():
            keep_d = torch.nonzero(d_global_alive, as_tuple=True)[0]
            new_d_nodes = self.dino_active_nodes[keep_d].contiguous()
            n_d = new_d_nodes.numel()
            new_fc_d = nn.Linear(768, n_d).to(device=device, dtype=dtype)
            new_fc_d.weight.data.copy_(self.filter_bank_dino.weight.data[keep_d])
            new_fc_d.bias.data.copy_(self.filter_bank_dino.bias.data[keep_d])
            self.filter_bank_dino = new_fc_d
            self.dino_active_nodes = new_d_nodes

        # 3. Trim dead Hop-1 outgoing edges from deleted nodes in the GNN
        self.gnn.trim_dead_hop1_edges(self.alive_mask)

    @torch.no_grad()
    def delete_subset_nodes(self, nodes_to_delete: torch.Tensor, trim_immediately: bool = False):
        newly_deleted = 0
        for idx in nodes_to_delete.tolist():
            if self.alive_mask[idx] > 0.5:
                self.alive_mask[idx] = 0.0
                newly_deleted += 1

        if trim_immediately and newly_deleted > 0:
            self.physically_trim_deleted_filters()
        else:
            # Zero out newly deleted rows in-place during sweep
            c_keep = (self.alive_mask[self.conv_active_nodes] > 0.5).float().view(-1, 1)
            d_keep = (self.alive_mask[1024 + self.dino_active_nodes] > 0.5).float().view(-1, 1)
            self.filter_bank_conv.weight.mul_(c_keep)
            self.filter_bank_conv.bias.mul_(c_keep.squeeze(-1))
            self.filter_bank_dino.weight.mul_(d_keep)
            self.filter_bank_dino.bias.mul_(d_keep.squeeze(-1))

        total_deleted = int((self.alive_mask < 0.5).sum().item())
        return newly_deleted, total_deleted

    @torch.no_grad()
    def edit_graph_delete_nodes(self, target_delete_ratio: float, node_order: torch.Tensor, trim_immediately: bool = False):
        target_deleted = int(round(target_delete_ratio * len(node_order)))
        if target_deleted <= 0:
            return 0, int((self.alive_mask < 0.5).sum().item())
        return self.delete_subset_nodes(node_order[:target_deleted], trim_immediately=trim_immediately)

    @torch.no_grad()
    def mask_deleted_filter_grads(self):
        c_keep = (self.alive_mask[self.conv_active_nodes] > 0.5).float().view(-1, 1)
        d_keep = (self.alive_mask[1024 + self.dino_active_nodes] > 0.5).float().view(-1, 1)
        if self.filter_bank_conv.weight.grad is not None:
            self.filter_bank_conv.weight.grad.mul_(c_keep)
        if self.filter_bank_conv.bias.grad is not None:
            self.filter_bank_conv.bias.grad.mul_(c_keep.squeeze(-1))
        if self.filter_bank_dino.weight.grad is not None:
            self.filter_bank_dino.weight.grad.mul_(d_keep)
        if self.filter_bank_dino.bias.grad is not None:
            self.filter_bank_dino.bias.grad.mul_(d_keep.squeeze(-1))

    @torch.no_grad()
    def enforce_zero_on_deleted_rows(self):
        c_keep = (self.alive_mask[self.conv_active_nodes] > 0.5).float().view(-1, 1)
        d_keep = (self.alive_mask[1024 + self.dino_active_nodes] > 0.5).float().view(-1, 1)
        self.filter_bank_conv.weight.mul_(c_keep)
        self.filter_bank_conv.bias.mul_(c_keep.squeeze(-1))
        self.filter_bank_dino.weight.mul_(d_keep)
        self.filter_bank_dino.bias.mul_(d_keep.squeeze(-1))

    def forward(self, f1_in, f2_in):
        B = f1_in.size(0)
        c_out = self.filter_bank_conv(f1_in)  # [B, len(conv_active_nodes)]
        d_out = self.filter_bank_dino(f2_in)  # [B, len(dino_active_nodes)]

        X_raw = torch.zeros(B, 1792, device=f1_in.device, dtype=f1_in.dtype)
        X_raw.index_copy_(1, self.conv_active_nodes, c_out)
        X_raw.index_copy_(1, 1024 + self.dino_active_nodes, d_out)
        X_raw = X_raw * self.alive_mask.view(1, -1)

        X_gnn = self.gnn(X_raw, self.alive_mask)            # [B, 1792]

        emb_conv = self.post_conv(X_gnn[:, :1024])          # [B, 1024]
        emb_dino = self.post_dino(X_gnn[:, 1024:])          # [B, 768]

        logits_conv = self.head_conv(emb_conv)              # [B, 200]
        logits_dino = self.head_dino(emb_dino)              # [B, 200]
        logits_ens = 0.5 * logits_conv + 0.5 * logits_dino
        emb_joint = torch.cat([emb_conv, emb_dino], dim=-1) # [B, 1792]

        return logits_conv, logits_dino, logits_ens, emb_conv, emb_dino, emb_joint


@torch.no_grad()
def evaluate_graph_editor(model, eval_dict, joint_probe, device, batch_size=256):
    model.eval()
    F1 = eval_dict["F1"]
    F2 = eval_dict["F2"]
    labels = eval_dict["labels"]
    n = labels.size(0)

    top1_c, top5_c = 0, 0
    top1_d, top5_d = 0, 0
    top1_e, top5_e = 0, 0
    top1_p, top5_p = 0, 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        f1_b = F1[start:end].to(device, non_blocking=True).float()
        f2_b = F2[start:end].to(device, non_blocking=True).float()
        y_b = labels[start:end].to(device, non_blocking=True)

        logits_c, logits_d, logits_e, _, _, emb_j = model(f1_b, f2_b)
        logits_p = joint_probe(emb_j)

        for logits, tag in [(logits_c, "c"), (logits_d, "d"), (logits_e, "e"), (logits_p, "p")]:
            _, pred5 = logits.topk(5, dim=-1, largest=True, sorted=True)
            corr5 = pred5.eq(y_b.view(-1, 1).expand_as(pred5))
            c1 = corr5[:, 0].sum().item()
            c5 = corr5[:, :5].sum().item()
            if tag == "c":
                top1_c += c1; top5_c += c5
            elif tag == "d":
                top1_d += c1; top5_d += c5
            elif tag == "e":
                top1_e += c1; top5_e += c5
            else:
                top1_p += c1; top5_p += c5

    return {
        "conv_top1": 100.0 * top1_c / n, "conv_top5": 100.0 * top5_c / n,
        "dino_top1": 100.0 * top1_d / n, "dino_top5": 100.0 * top5_d / n,
        "ens_top1":  100.0 * top1_e / n, "ens_top5":  100.0 * top5_e / n,
        "probe_top1": 100.0 * top1_p / n, "probe_top5": 100.0 * top5_p / n,
    }


def train_one_graph_epoch(model, train_loader, optimizer, temp, device):
    model.train()
    total_loss = 0.0
    total = 0

    for f1_b, f2_b, y_b, t1_l, t2_l in train_loader:
        f1_b = f1_b.to(device, non_blocking=True).float()
        f2_b = f2_b.to(device, non_blocking=True).float()
        y_b = y_b.to(device, non_blocking=True)
        t1_l = t1_l.to(device, non_blocking=True).float()
        t2_l = t2_l.to(device, non_blocking=True).float()

        logits_c, logits_d, _, emb_c, emb_d, _ = model(f1_b, f2_b)

        ce_c = F.cross_entropy(logits_c, y_b, label_smoothing=0.1)
        kl_c_from_dino = F.kl_div(
            F.log_softmax(logits_c / temp, dim=-1),
            F.softmax(t2_l / temp, dim=-1),
            reduction="batchmean",
        ) * (temp ** 2)
        kl_c_self = F.kl_div(
            F.log_softmax(logits_c / temp, dim=-1),
            F.softmax(t1_l / temp, dim=-1),
            reduction="batchmean",
        ) * (temp ** 2)
        feat_c = (1.0 - F.cosine_similarity(emb_c, f1_b, dim=-1)).mean() + 2.0 * F.smooth_l1_loss(emb_c, f1_b)
        loss_c = 0.3 * ce_c + 0.4 * kl_c_from_dino + 0.3 * kl_c_self + 2.0 * feat_c

        ce_d = F.cross_entropy(logits_d, y_b, label_smoothing=0.1)
        kl_d_from_conv = F.kl_div(
            F.log_softmax(logits_d / temp, dim=-1),
            F.softmax(t1_l / temp, dim=-1),
            reduction="batchmean",
        ) * (temp ** 2)
        kl_d_self = F.kl_div(
            F.log_softmax(logits_d / temp, dim=-1),
            F.softmax(t2_l / temp, dim=-1),
            reduction="batchmean",
        ) * (temp ** 2)
        feat_d = (1.0 - F.cosine_similarity(emb_d, f2_b, dim=-1)).mean() + 2.0 * F.smooth_l1_loss(emb_d, f2_b)
        loss_d = 0.3 * ce_d + 0.4 * kl_d_from_conv + 0.3 * kl_d_self + 2.0 * feat_d

        loss = 0.5 * (loss_c + loss_d)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        model.mask_deleted_filter_grads()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        model.enforce_zero_on_deleted_rows()

        bs = y_b.size(0)
        total_loss += loss.item() * bs
        total += bs

    return total_loss / total


def make_optimizer(model, lr_scale=1.0):
    return torch.optim.AdamW(
        [
            {"params": list(model.filter_bank_conv.parameters()) + list(model.filter_bank_dino.parameters()), "lr": 8e-4 * lr_scale, "weight_decay": 1e-4},
            {"params": list(model.gnn.parameters()), "lr": 2e-3 * lr_scale, "weight_decay": 0.0},
            {"params": list(model.post_conv.parameters()) + list(model.post_dino.parameters()), "lr": 8e-4 * lr_scale, "weight_decay": 1e-4},
            {"params": list(model.head_conv.parameters()) + list(model.head_dino.parameters()), "lr": 1e-4 * lr_scale, "weight_decay": 1e-4},
        ]
    )


def save_trimmed_checkpoint(model: DualBackboneGraphEditor, path: str, epoch: int, val_metrics: dict):
    """
    Physically trims all deleted filter rows and dead Hop-1 outgoing edges from a copy of `model`
    and saves the genuinely trimmed checkpoint to `path`!
    """
    trimmed_model = copy.deepcopy(model)
    trimmed_model.physically_trim_deleted_filters()
    trimmed_params = trimmed_model.count_total_params()
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": trimmed_model.state_dict(),
            "conv_active_nodes": trimmed_model.conv_active_nodes.cpu(),
            "dino_active_nodes": trimmed_model.dino_active_nodes.cpu(),
            "alive_mask": trimmed_model.alive_mask.cpu(),
            "trimmed_total_params": trimmed_params,
            "val_metrics": val_metrics,
        },
        path,
    )
    return trimmed_model, trimmed_params


def run_pareto_two_stage(
    model: DualBackboneGraphEditor,
    train_loader,
    val_test_data,
    joint_probe,
    redundancy_order,
    args,
    device,
):
    dino_nodes = redundancy_order[redundancy_order >= 1024]
    conv_nodes = redundancy_order[redundancy_order < 1024]

    p1_epochs = 45
    p2_epochs = 45
    p3_epochs = 15
    max_sweep_ratio = 0.85
    min_pareto_ratio = args.min_delete

    pre_trim_params = model.count_total_params()
    print("\n" + "=" * 84)
    print(f" PHASE 1: DINOv2 PARETO SWEEP (0% -> {max_sweep_ratio*100:.0f}% DINO Deletion | Pre-Trim Params: {pre_trim_params/1e6:.3f}M)")
    print("=" * 84)

    opt1 = make_optimizer(model, lr_scale=1.0)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=p1_epochs, eta_min=2e-4)

    best_dino_val = -1.0
    best_dino_state = None
    best_dino_del_count = 0
    t0 = time.time()

    for ep in range(1, p1_epochs + 1):
        dino_del_ratio = 0.0 if ep < 3 else max_sweep_ratio * ((ep - 2) / float(p1_epochs - 2))
        model.edit_graph_delete_nodes(dino_del_ratio, dino_nodes)

        loss = train_one_graph_epoch(model, train_loader, opt1, args.temp, device)
        sched1.step()

        c_alive = int(model.alive_mask[:1024].sum().item())
        d_alive = int(model.alive_mask[1024:].sum().item())
        d_del = 768 - d_alive
        d_del_pct = 100.0 * d_del / 768.0

        val_m = evaluate_graph_editor(model, val_test_data["validation"], joint_probe, device)
        is_pareto_peak = False
        if (d_del / 768.0) >= min_pareto_ratio and val_m["dino_top1"] > best_dino_val:
            best_dino_val = val_m["dino_top1"]
            best_dino_state = copy.deepcopy(model.state_dict())
            best_dino_del_count = d_del
            is_pareto_peak = True

        if ep % 3 == 0 or ep == 1 or is_pareto_peak or ep == p1_epochs:
            peak_tag = " << NEW DINO PARETO PEAK!" if is_pareto_peak else ""
            print(
                f"[P1 Ep {ep:2d}/{p1_epochs}] DINO Deleted: {d_del_pct:5.1f}% ({d_del:3d}/768 | Alive: Conv={c_alive}, DINO={d_alive:3d}) | "
                f"Val: [Conv: {val_m['conv_top1']:.2f}% | DINO: {val_m['dino_top1']:.2f}% | Ens: {val_m['ens_top1']:.2f}%] | "
                f"{time.time()-t0:.1f}s{peak_tag}"
            )

    # Restore Phase-1 Pareto-Optimal DINOv2 Subgraph AND Physically Trim DINOv2 Filter Bank!
    model.load_state_dict(best_dino_state)
    model.physically_trim_deleted_filters()
    p1_trimmed_params = model.count_total_params()
    d_alive_locked = int(model.alive_mask[1024:].sum().item())

    print("\n" + "-" * 84)
    print(
        f">>> LOCKED & PHYSICALLY TRIMMED PHASE-1 DINOv2 SUBGRAPH:\n"
        f"    DINOv2 Filter Bank Trimmed: [768, 768] -> {list(model.filter_bank_dino.weight.shape)} "
        f"({best_dino_del_count}/768 Deleted = {100.0*best_dino_del_count/768:.1f}% Removed!)\n"
        f"    Layer Parameters: {pre_trim_params/1e6:.3f}M -> {p1_trimmed_params/1e6:.3f}M | Peak DINO Val = {best_dino_val:.2f}%"
    )
    print("-" * 84)

    print("\n" + "=" * 84)
    print(f" PHASE 2: CONVNEXT PARETO SWEEP (0% -> {max_sweep_ratio*100:.0f}% ConvNeXt Deletion over {p2_epochs} Epochs)")
    print("=" * 84)

    opt2 = make_optimizer(model, lr_scale=0.8)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=p2_epochs, eta_min=1e-4)

    best_p2_score = -1.0
    best_p2_state = None
    best_conv_del_count = 0

    for ep in range(1, p2_epochs + 1):
        conv_del_ratio = 0.0 if ep < 3 else max_sweep_ratio * ((ep - 2) / float(p2_epochs - 2))
        model.edit_graph_delete_nodes(conv_del_ratio, conv_nodes)

        loss = train_one_graph_epoch(model, train_loader, opt2, args.temp, device)
        sched2.step()

        c_alive = int(model.alive_mask[:1024].sum().item())
        d_alive = int(model.alive_mask[1024:].sum().item())
        c_del = 1024 - c_alive
        c_del_pct = 100.0 * c_del / 1024.0

        val_m = evaluate_graph_editor(model, val_test_data["validation"], joint_probe, device)
        score = 0.5 * (val_m["ens_top1"] + val_m["conv_top1"])
        is_pareto_peak = False
        if (c_del / 1024.0) >= min_pareto_ratio and score > best_p2_score:
            best_p2_score = score
            best_p2_state = copy.deepcopy(model.state_dict())
            best_conv_del_count = c_del
            is_pareto_peak = True

        if ep % 3 == 0 or ep == 1 or is_pareto_peak or ep == p2_epochs:
            peak_tag = " << NEW JOINT PARETO PEAK!" if is_pareto_peak else ""
            print(
                f"[P2 Ep {ep:2d}/{p2_epochs}] Conv Deleted: {c_del_pct:5.1f}% ({c_del:3d}/1024 | Alive: Conv={c_alive:3d}, DINO={d_alive:3d}) | "
                f"Val: [Conv: {val_m['conv_top1']:.2f}% | DINO: {val_m['dino_top1']:.2f}% | Ens: {val_m['ens_top1']:.2f}%] | "
                f"{time.time()-t0:.1f}s{peak_tag}"
            )

    # Restore Phase-2 Joint Pareto-Optimal Subgraph AND Physically Trim ConvNeXt Filter Bank & GNN!
    model.load_state_dict(best_p2_state)
    model.physically_trim_deleted_filters()
    p2_trimmed_params = model.count_total_params()
    c_alive_locked = int(model.alive_mask[:1024].sum().item())
    total_del_locked = int((model.alive_mask < 0.5).sum().item())

    print("\n" + "-" * 84)
    print(
        f">>> LOCKED & PHYSICALLY TRIMMED JOINT PARETO-OPTIMAL SUBGRAPH:\n"
        f"    ConvNeXt Filter Bank Trimmed : [1024, 1024] -> {list(model.filter_bank_conv.weight.shape)}\n"
        f"    DINOv2 Filter Bank Trimmed   : [768, 768]   -> {list(model.filter_bank_dino.weight.shape)}\n"
        f"    GNN Hop-1 Active Edges       : {model.gnn.edge_weights[0].numel():,d} (trimmed from {model.gnn.edge_weights[1].numel():,d})\n"
        f"    Total Layer Parameters       : {pre_trim_params/1e6:.3f}M (Pre-Trim) -> {p2_trimmed_params/1e6:.3f}M (Post-Trim = {pre_trim_params/p2_trimmed_params:.2f}x smaller!)"
    )
    print("-" * 84)

    print(f"\n[Phase 3] Fine-Tuning Physically Trimmed Subgraph ({p2_trimmed_params/1e6:.3f}M Params | {1792-total_del_locked}/1792 Filters) for {p3_epochs} Epochs...")
    opt3 = make_optimizer(model, lr_scale=0.4)
    sched3 = torch.optim.lr_scheduler.CosineAnnealingLR(opt3, T_max=p3_epochs, eta_min=1e-5)
    best_final_ens = -1.0
    best_trimmed_model = None

    for ep in range(1, p3_epochs + 1):
        loss = train_one_graph_epoch(model, train_loader, opt3, args.temp, device)
        sched3.step()
        val_m = evaluate_graph_editor(model, val_test_data["validation"], joint_probe, device)
        if val_m["ens_top1"] > best_final_ens:
            best_final_ens = val_m["ens_top1"]
            best_trimmed_model, _ = save_trimmed_checkpoint(model, args.save_ckpt, ep, val_m)
        if ep % 5 == 0 or ep == p3_epochs:
            print(
                f"[P3 Ep {ep:2d}/{p3_epochs}] Trimmed Params: {p2_trimmed_params/1e6:.3f}M (Conv={c_alive_locked}, DINO={d_alive_locked}) | "
                f"Val: [Conv: {val_m['conv_top1']:.2f}% | DINO: {val_m['dino_top1']:.2f}% | "
                f"Ens: {val_m['ens_top1']:.2f}% (Top-5: {val_m['ens_top5']:.2f}%)]"
            )

    return best_trimmed_model


def get_graph_edit_schedule(epoch: int, max_delete: float = 0.50) -> float:
    if epoch < 3:
        return 0.0
    elif epoch <= 72:
        progress = (epoch - 2) / 70.0
        return max_delete * progress
    else:
        return max_delete


def main():
    parser = argparse.ArgumentParser(description="KD as Cross-Backbone Graph Editing via Cumulative MFH")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-delete", type=float, default=0.50)
    parser.add_argument("--min-delete", type=float, default=0.40)
    parser.add_argument(
        "--order",
        type=str,
        default="pareto_two_stage",
        choices=["pareto_two_stage", "redundancy", "random", "delete_dino", "delete_conv"],
    )
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--temp", type=float, default=2.0)
    parser.add_argument("--save-ckpt", type=str, default="graph_edited_dual_backbone.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.order == "delete_dino":
        args.max_delete = 768.0 / 1792.0
    elif args.order == "delete_conv":
        args.max_delete = 1024.0 / 1792.0

    print("=" * 84)
    print(" CROSS-BACKBONE GRAPH EDITING WITH PHYSICAL FILTER TRIMMING")
    print(f" Device: {device} | Mode: {args.order} | Batch Size: {args.batch_size}")
    print("=" * 84)

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
        top_k=args.top_k,
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

    pre_trim_params = model.count_total_params()
    val0 = evaluate_graph_editor(model, val_test_data["validation"], joint_probe, device)
    test0 = evaluate_graph_editor(model, val_test_data["test"], joint_probe, device)
    print(
        f"\n>>> STEP-0 PRE-TRIM ({pre_trim_params/1e6:.3f}M Params | 0% Nodes Deleted | 1792/1792 Alive):\n"
        f"    VAL  -> [ConvNeXt: {val0['conv_top1']:.2f}% | DINOv2: {val0['dino_top1']:.2f}% | "
        f"Ensemble: {val0['ens_top1']:.2f}% (Top-5: {val0['ens_top5']:.2f}%)]\n"
        f"    TEST -> [ConvNeXt: {test0['conv_top1']:.2f}% | DINOv2: {test0['dino_top1']:.2f}% | "
        f"Ensemble: {test0['ens_top1']:.2f}% (Top-5: {test0['ens_top5']:.2f}%)]"
    )

    train_loader = FastBatchLoader(
        F1_train, F2_train, y_train, t1_logits_train, t2_logits_train,
        batch_size=args.batch_size, shuffle=True
    )

    if args.order == "pareto_two_stage":
        eval_model = run_pareto_two_stage(
            model=model,
            train_loader=train_loader,
            val_test_data=val_test_data,
            joint_probe=joint_probe,
            redundancy_order=redundancy_order,
            args=args,
            device=device,
        )
    else:
        if args.order == "redundancy":
            node_order = redundancy_order
        elif args.order == "delete_dino":
            dino_nodes = redundancy_order[redundancy_order >= 1024]
            conv_nodes = redundancy_order[redundancy_order < 1024]
            node_order = torch.cat([dino_nodes, conv_nodes], dim=0)
        elif args.order == "delete_conv":
            conv_nodes = redundancy_order[redundancy_order < 1024]
            dino_nodes = redundancy_order[redundancy_order >= 1024]
            node_order = torch.cat([conv_nodes, dino_nodes], dim=0)
        else:
            gen = torch.Generator()
            gen.manual_seed(args.seed)
            node_order = torch.randperm(1792, generator=gen)

        optimizer = make_optimizer(model, lr_scale=1.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
        best_full_edit_val = 0.0
        eval_model = None
        t_start = time.time()

        for epoch in range(1, args.epochs + 1):
            del_ratio = get_graph_edit_schedule(epoch, max_delete=args.max_delete)
            _, total_del = model.edit_graph_delete_nodes(del_ratio, node_order)
            c_alive = int(model.alive_mask[:1024].sum().item())
            d_alive = int(model.alive_mask[1024:].sum().item())

            # At Epoch 72 (when target deletion ratio is reached), physically trim the deleted filter rows!
            if epoch == 72:
                model.physically_trim_deleted_filters()
                optimizer = make_optimizer(model, lr_scale=0.35)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - 71, eta_min=1e-5)
                print(
                    f"\n>>> [Epoch 72] PHYSICALLY TRIMMED FILTER BANKS: "
                    f"Conv={list(model.filter_bank_conv.weight.shape)}, DINO={list(model.filter_bank_dino.weight.shape)} | "
                    f"Params: {pre_trim_params/1e6:.3f}M -> {model.count_total_params()/1e6:.3f}M!"
                )

            avg_loss = train_one_graph_epoch(model, train_loader, optimizer, args.temp, device)
            scheduler.step()

            if epoch % 5 == 0 or epoch == 1 or epoch == 72 or epoch == args.epochs:
                val_m = evaluate_graph_editor(model, val_test_data["validation"], joint_probe, device)
                elapsed = time.time() - t_start
                print(
                    f"Epoch [{epoch:3d}/{args.epochs}] | Params: {model.count_total_params()/1e6:.3f}M | "
                    f"Deleted: {100.0*total_del/1792:4.1f}% ({total_del:3d}/1792 | Alive: Conv={c_alive}, DINO={d_alive}) | "
                    f"Val: [Conv: {val_m['conv_top1']:.2f}% | DINO: {val_m['dino_top1']:.2f}% | "
                    f"Ens: {val_m['ens_top1']:.2f}% (Top-5: {val_m['ens_top5']:.2f}%)] | {elapsed:.1f}s"
                )

            if epoch >= 72:
                val_m = evaluate_graph_editor(model, val_test_data["validation"], joint_probe, device)
                if val_m["ens_top1"] > best_full_edit_val:
                    best_full_edit_val = val_m["ens_top1"]
                    eval_model, _ = save_trimmed_checkpoint(model, args.save_ckpt, epoch, val_m)

    # Final Evaluation on the Physically Trimmed Model
    val_final = evaluate_graph_editor(eval_model, val_test_data["validation"], joint_probe, device)
    test_final = evaluate_graph_editor(eval_model, val_test_data["test"], joint_probe, device)
    post_trim_params = eval_model.count_total_params()
    total_del = int((eval_model.alive_mask < 0.5).sum().item())
    c_alive = eval_model.conv_active_nodes.numel()
    d_alive = eval_model.dino_active_nodes.numel()

    print("\n" + "=" * 84)
    print(f" FINAL PHYSICALLY TRIMMED BENCHMARK ({100.0*total_del/1792:.1f}% NODES DELETED | {1792-total_del}/1792 ALIVE)")
    print(f" Trimmed Filter Bank Shapes : ConvNeXt = {list(eval_model.filter_bank_conv.weight.shape)} | DINOv2 = {list(eval_model.filter_bank_dino.weight.shape)}")
    print(f" Layer Parameter Count      : {pre_trim_params:,d} ({pre_trim_params/1e6:.3f}M Pre-Trim) -> {post_trim_params:,d} ({post_trim_params/1e6:.3f}M Post-Trim = {pre_trim_params/post_trim_params:.2f}x smaller!)")
    print("=" * 84)
    print(f"  [VALIDATION] ConvNeXt: {val_final['conv_top1']:.2f}% (Top-5: {val_final['conv_top5']:.2f}%) | "
          f"DINOv2: {val_final['dino_top1']:.2f}% (Top-5: {val_final['dino_top5']:.2f}%) | "
          f"Ensemble: {val_final['ens_top1']:.2f}% (Top-5: {val_final['ens_top5']:.2f}%)")
    print(f"  [TEST]       ConvNeXt: {test_final['conv_top1']:.2f}% (Top-5: {test_final['conv_top5']:.2f}%) | "
          f"DINOv2: {test_final['dino_top1']:.2f}% (Top-5: {test_final['dino_top5']:.2f}%) | "
          f"Ensemble: {test_final['ens_top1']:.2f}% (Top-5: {test_final['ens_top5']:.2f}%)")
    print("=" * 84 + "\n")


if __name__ == "__main__":
    main()
