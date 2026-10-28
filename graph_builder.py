import torch
import torch.nn as nn

class HomophilyGraphBuilder(nn.Module):
    def __init__(self, threshold=0.30, local_threshold=0.35, prune_orphans=True):
        super().__init__()
        self.threshold = threshold
        self.local_threshold = local_threshold
        self.prune_orphans = prune_orphans

    def forward(self, f1, f2):
        # f1: [B, D1], f2: [B, D2]
        B, D1 = f1.shape
        _, D2 = f2.shape
        N = D1 + D2

        # 1. Full Pearson correlation matrix [N, N] across all features
        F = torch.cat([f1, f2], dim=1)
        F_c = F - F.mean(dim=0, keepdim=True)
        std = F_c.norm(dim=0, keepdim=True).clamp(min=1e-6)
        corr = (F_c.T @ F_c) / (std.T @ std)

        # 2. Partition into 3 communities: Global Core, BB1 Local, BB2 Local
        cross = corr[:D1, D1:].abs()
        core1 = (cross.max(dim=1).values >= self.threshold).nonzero().squeeze(-1)
        core2 = D1 + (cross.max(dim=0).values >= self.threshold).nonzero().squeeze(-1)
        core_nodes = torch.cat([core1, core2])

        local1 = (cross.max(dim=1).values < self.threshold).nonzero().squeeze(-1)
        local2 = D1 + (cross.max(dim=0).values < self.threshold).nonzero().squeeze(-1)

        # 3. Prune isolated orphan nodes with zero internal connections
        if self.prune_orphans:
            if len(local1) > 0:
                survived1 = (corr[local1][:, local1].abs() >= self.local_threshold).sum(dim=1) > 1
                local1 = local1[survived1]
            if len(local2) > 0:
                survived2 = (corr[local2][:, local2].abs() >= self.local_threshold).sum(dim=1) > 1
                local2 = local2[survived2]

        # 4. Build Adjacency Matrix with intra-community homophily
        adj = torch.eye(N, device=f1.device)
        for group, thresh in [(core_nodes, self.threshold), (local1, self.local_threshold), (local2, self.local_threshold)]:
            if len(group) > 0:
                sub_corr = corr[group][:, group].abs()
                mask = sub_corr >= thresh
                adj[group[:, None], group] = torch.where(mask, sub_corr, adj[group[:, None], group])

        # 5. Connect communities via single strongest Bridge Edges
        bridges = []
        for local_group in [local1, local2]:
            if len(local_group) > 0 and len(core_nodes) > 0:
                sub = corr[local_group][:, core_nodes].abs()
                idx_local, idx_core = (sub == sub.max()).nonzero()[0]
                u, v = local_group[idx_local].item(), core_nodes[idx_core].item()
                adj[u, v] = adj[v, u] = sub[idx_local, idx_core]
                bridges.append((u, v, adj[u, v].item()))

        return adj, core_nodes, local1, local2, bridges
