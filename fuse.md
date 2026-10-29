# Multi-Backbone Feature Fusion Taxonomy & LERP Formulations
## Grounding Representation Merging in Topological Bridge Interfaces

---

### 1. Motivation: Beyond Naive Concatenation

In early baseline implementations, modular student fusion relied on naive scalar gating and channel concatenation:
```python
fused = torch.cat([s_core, alpha[:, 0:1] * s_sat1, alpha[:, 1:2] * s_sat2], dim=-1)
```
While functional, this approach suffers from two limitations:
1. **Unbounded Dimension Growth**: Concatenation expands the feature representation linearly with each new backbone ($D = D_{core} + \sum D_{sat_k}$), diluting classifier weights.
2. **Topological Disconnect**: It ignores the **Heterophilic Bridge Edges** discovered in Stage 1, treating the private specialist islands as isolated appendages rather than networks anchored to the shared core.

By re-framing fusion as **Linear Interpolation (LERP)** at the bridge boundaries, we create a continuous, smooth transition between invariant visual primitives and specialized backbone inductive biases.

---

### 2. The K-Backbone Symmetry Problem

A naive 1D vector boundary (e.g., placing ResNet at index `0..31` and Swin at index `96..127`) introduces an arbitrary coordinate bias:
- Why should ResNet occupy the "left" and Swin occupy the "right"?
- If a 3rd or 4th backbone is introduced (e.g., ConvNeXt or DINOv2), a 1D vector runs out of "ends".

To solve this, we define two mathematically symmetric, order-invariant LERP strategies for any arbitrary number of backbones $K \ge 2$.

---

### 3. Strategy 1: Slotted Radial LERP (Order-Invariant Partitioning)

#### 3.1 Conceptual Blueprint
The Core representation $s_{core} \in \mathbb{R}^{D_{core}}$ is partitioned into an **Invariant Base** (untouched consensus) plus $K$ dedicated **Anchor Slots** (one for each heterogeneous backbone):

```text
                           ┌──────────────────────────────────────────────┐
                           │              Core Invariant Hub              │
                           │  [128 dims = 64 Invariant Base + 2 x 32 Slots│
                           └──────┬───────────────────────────────┬───────┘
                                  │                               │
                       Core Slot 1 [32 dims]           Core Slot 2 [32 dims]
                                  │                               │
                            LERP(α₁)                        LERP(α₂)
                                  │                               │
                        Sat 1 [32 dims]                 Sat 2 [32 dims]
                        (ResNet DW-Conv)                (Swin Attention)
```

#### 3.2 Mathematical Formulation
For each backbone $k \in \{1, \dots, K\}$:
$$\text{slot}_k^{fused} = \text{lerp}(s_{core}^{(k)}, s_{sat}^{(k)}, \alpha_k) = (1 - \alpha_k) \cdot s_{core}^{(k)} + \alpha_k \cdot s_{sat}^{(k)}$$

The final representation concatenates the untouched invariant base with the interpolated slots:
$$s_{fused} = \left[ s_{core}^{base}, \; \text{slot}_1^{fused}, \; \text{slot}_2^{fused}, \; \dots, \; \text{slot}_K^{fused} \right]$$

#### 3.3 PyTorch Implementation
```python
class SlottedLerpFusion(nn.Module):
    def __init__(self, core_dim=128, num_backbones=2, slot_dim=32):
        super().__init__()
        self.num_backbones = num_backbones
        self.slot_dim = slot_dim
        self.base_dim = core_dim - (num_backbones * slot_dim)  # e.g., 128 - 64 = 64
        assert self.base_dim > 0, "Core dimension must be larger than total slot dimensions."

    def forward(self, s_core, sat_list, alphas):
        """
        s_core: [B, D_core]
        sat_list: list of K tensors, each [B, slot_dim]
        alphas: [B, K] gating coefficients in [0, 1]
        """
        # 1. Slice core into invariant base and K anchor slots
        core_base = s_core[:, :self.base_dim]
        fused_slots = []

        for k in range(self.num_backbones):
            start = self.base_dim + k * self.slot_dim
            end = start + self.slot_dim
            core_slot_k = s_core[:, start:end]
            sat_k = sat_list[k]
            a_k = alphas[:, k:k+1]

            # Linear Interpolation
            blended = torch.lerp(core_slot_k, sat_k, a_k)
            fused_slots.append(blended)

        return torch.cat([core_base] + fused_slots, dim=-1)
```

**Pros**:
- Strictly order-invariant across all $K$ backbones.
- Preserves a clean sub-vector that is 100% invariant consensus.

---

### 4. Strategy 2: Residual Graph Bridge LERP (Dimension-Preserving)

#### 4.1 Algebraic Insight
Linear interpolation can be rewritten as a residual modulation:
$$\text{lerp}(A, B, \alpha) = A + \alpha \cdot (B - A)$$
Here:
- $A$ is the **Core Hub** (invariant semantic foundation).
- $B$ is the **Specialist Island** (projected into Core space via the maximal-affinity Bridge Edge).
- $(B - A)$ is the **Residual Inductive Delta** that the specialist adds over the core.
- $\alpha \in [0, 1]$ dynamically meters how much of this delta is infused.

```text
                  s_core [128 dims]
                        │
                        ▼
                ┌───────────────┐
      Sat 1 ───►│ Bridge Proj 1 ├───────┐
                └───────────────┘       ▼
                                   [ + α₁ · (Sat₁ - Core) ]
                ┌───────────────┐       ▲
      Sat 2 ───►│ Bridge Proj 2 ├───────┘
                └───────────────┘       │
                                        ▼
                                s_fused [128 dims] (Zero Dimension Growth!)
```

#### 4.2 Mathematical Formulation
For each backbone $k \in \{1, \dots, K\}$, the specialist representation $s_{sat}^{(k)}$ is projected to the core dimension via the bridge adapter $W_{bridge}^{(k)}$:
$$\tilde{s}_{sat}^{(k)} = W_{bridge}^{(k)} s_{sat}^{(k)}$$

The fused representation is formed by accumulating the gated residual LERP deltas:
$$s_{fused} = s_{core} + \sum_{k=1}^K \alpha_k \cdot \left( \tilde{s}_{sat}^{(k)} - s_{core} \right)$$

#### 4.3 PyTorch Implementation (Base Form: Independent Gating)
```python
class ResidualBridgeLerpFusion(nn.Module):
    def __init__(self, core_dim=128, sat_dims=[32, 16]):
        super().__init__()
        # Bridge projection adapters mapping each specialist into Core manifold
        self.bridge_adapters = nn.ModuleList([
            nn.Linear(dim, core_dim, bias=False) for dim in sat_dims
        ])

    def forward(self, s_core, sat_list, alphas):
        """
        s_core: [B, core_dim]
        sat_list: list of K tensors [B, sat_dim_k]
        alphas: [B, K] in [0, 1] (independent sigmoids)
        """
        fused = s_core.clone()
        for k, (sat, adapter) in enumerate(zip(sat_list, self.bridge_adapters)):
            sat_in_core = adapter(sat)
            a_k = alphas[:, k:k+1]
            # Accumulate residual LERP delta
            fused = fused + a_k * (sat_in_core - s_core)

        return fused  # Shape remains [B, core_dim]!
```

---

#### 4.4 The "Feature Soup" Pathology with Many Backbones ($K \gg 2$)
While independent residual addition works cleanly for $K = 2$, stacking $5$ to $10$ backbones using independent sigmoids creates **Representation Collapse ("Feature Soup")**:
1. **Norm / Variance Inflation**: Blending $10$ unconstrained deltas inflates the norm $\|s_{fused}\|$, drowning out the invariant core.
2. **Gradient Tug-of-War**: Conflicting inductive biases across multiple backbones cancel each other into unstructured noise.

To scale Bridge LERP to arbitrary numbers of teachers without feature soup, we introduce two specialized routing variations:

---

#### 4.5 Variation 2A: The Convex Gate (Softmax Interpolation Hull)

##### Concept & Formulation
Instead of unconstrained independent activations, the router outputs a normalized probability distribution over all components (Core + $K$ Teacher Satellites) using `Softmax`:
$$[\alpha_0, \; \alpha_1, \; \dots, \; \alpha_K] = \text{Softmax}\left( \text{Router}(s_{trunk}) \right), \quad \sum_{i=0}^K \alpha_i = 1.0$$

The fused embedding is strictly guaranteed to lie inside the **Convex Hull** of the teachers and core, preventing norm explosion:
$$s_{fused} = \alpha_0 \cdot s_{core} + \sum_{k=1}^K \alpha_k \cdot \tilde{s}_{sat}^{(k)} = s_{core} + \sum_{k=1}^K \alpha_k \cdot \left( \tilde{s}_{sat}^{(k)} - s_{core} \right)$$

##### PyTorch Implementation
```python
class ConvexBridgeLerpFusion(nn.Module):
    """
    Variation 2A: Constrains the fused representation to the convex hull
    of the Core and K Teacher Satellites via Softmax routing. Prevents feature soup.
    """
    def __init__(self, core_dim=128, sat_dims=[32, 16]):
        super().__init__()
        self.num_backbones = len(sat_dims)
        self.bridge_adapters = nn.ModuleList([
            nn.Linear(dim, core_dim, bias=False) for dim in sat_dims
        ])

    def forward(self, s_core, sat_list, router_logits):
        """
        s_core:         [B, core_dim]
        sat_list:       list of K tensors [B, sat_dim_k]
        router_logits:  [B, K + 1] unnormalized logits for [Core, Sat_1, ..., Sat_K]
        """
        # Softmax over (Core + K Satellites) -> sum(weights) == 1.0
        weights = F.softmax(router_logits, dim=-1)  # [B, K + 1]
        w_core = weights[:, 0:1]                    # [B, 1]

        fused = w_core * s_core
        for k, (sat, adapter) in enumerate(zip(sat_list, self.bridge_adapters)):
            sat_in_core = adapter(sat)
            w_k = weights[:, k+1:k+2]
            fused = fused + w_k * sat_in_core

        return fused  # Shape remains [B, core_dim] with strictly bounded variance!
```

---

#### 4.6 Variation 2B: Sparse Routing (Mixture of Teachers / Top-K)

##### Concept & Formulation
For a large suite of teachers ($K = 5 \dots 10+$), most teachers are irrelevant for any single specific image. Sparse routing dynamically selects only the **Top-$M$ most confident teachers** (e.g., $M = 1$ or $M = 2$) per sample, setting all other $\alpha_k = 0$:
- High-frequency texture images dynamically route exclusively to ResNet ($M=1$).
- Broad global composition images dynamically route exclusively to Swin ($M=1$).
- The remaining 8 backbones are masked out completely, eliminating interference.

##### PyTorch Implementation
```python
class SparseTopKBridgeLerpFusion(nn.Module):
    """
    Variation 2B: Mixture-of-Teachers with sparse Top-M gating.
    Each sample queries only the top M most relevant teacher bridges.
    """
    def __init__(self, core_dim=128, sat_dims=[32, 16], top_m=1):
        super().__init__()
        self.top_m = top_m
        self.num_backbones = len(sat_dims)
        self.bridge_adapters = nn.ModuleList([
            nn.Linear(dim, core_dim, bias=False) for dim in sat_dims
        ])

    def forward(self, s_core, sat_list, router_logits):
        """
        s_core:         [B, core_dim]
        sat_list:       list of K tensors [B, sat_dim_k]
        router_logits:  [B, K] logits predicting teacher relevance
        """
        B, K = router_logits.shape
        # Select top-M teachers per sample
        top_logits, top_indices = torch.topk(router_logits, k=self.top_m, dim=-1)
        top_weights = F.softmax(top_logits, dim=-1)  # [B, top_m]

        # Create sparse weight tensor
        sparse_weights = torch.zeros_like(router_logits).scatter_(-1, top_indices, top_weights)

        # Apply residual LERP delta only from active teachers
        fused = s_core.clone()
        for k, (sat, adapter) in enumerate(zip(sat_list, self.bridge_adapters)):
            w_k = sparse_weights[:, k:k+1]
            if (w_k > 0).any():  # Active for at least one batch item
                sat_in_core = adapter(sat)
                fused = fused + w_k * (sat_in_core - s_core)

        return fused  # Zero soup, strict specialization!
```

---

### 5. Strategy 3: Boundary Interface & Point-Contact LERP (Fixed 128-dim Core)

#### 5.1 Conceptual Blueprint
This strategy keeps the core embedding at exactly 128 dimensions, anchoring the specialist backbones to the boundary extrema (the 0-th and 127-th indices of `z_core`):
- **Left Contact**: ResNet satellite (32 dims, indexed `0..31`) has its final channel (`index 31`) touch the core at `index 0`.
- **Right Contact**: Swin satellite (16 dims, indexed `0..15`) has its initial channel (`index 0`) touch the core at `index 127`.
- **Contact LERP**: Only the contact points are interpolated, leaving the interior (`1..126`) as pure core consensus, or optionally blending across a small boundary window:

```text
  ResNet Satellite [32 dims]                Core Invariant Hub [128 dims]               Swin Satellite [16 dims]
 [ 0 ................. 31 ]              [ 0 ............................ 127 ]            [ 0 ................. 15 ]
                        │                  │                                │                │
                        └── LERP(α₁) ──────┘                                └──── LERP(α₂) ──┘
                             Index 0                                            Index 127
                     (ResNet[31] + Core[0])                              (Core[127] + Swin[0])
```

#### 5.2 Mathematical Formulation
For point-contact interpolation:
$$s_{fused}[0] = \text{lerp}(s_{core}[0], s_{resnet}[31], \alpha_1) = (1 - \alpha_1) \cdot s_{core}[0] + \alpha_1 \cdot s_{resnet}[31]$$
$$s_{fused}[127] = \text{lerp}(s_{core}[127], s_{swin}[0], \alpha_2) = (1 - \alpha_2) \cdot s_{core}[127] + \alpha_2 \cdot s_{swin}[0]$$
$$s_{fused}[1:126] = s_{core}[1:126]$$

For boundary-window interpolation (e.g., span of $W$ channels at each flank):
$$s_{fused}[0:W] = \text{lerp}(s_{core}[0:W], s_{resnet}[32-W:32], \alpha_1)$$
$$s_{fused}[128-W:128] = \text{lerp}(s_{core}[128-W:128], s_{swin}[0:W], \alpha_2)$$

#### 5.3 PyTorch Implementation
```python
class PointContactBoundaryLerpFusion(nn.Module):
    """
    Interpolates specialist satellites at the boundary edges of the 128-dim core.
    Supports point-contact (window=1) or boundary window blending (window=W).
    """
    def __init__(self, core_dim=128, window=1):
        super().__init__()
        self.core_dim = core_dim
        self.window = window

    def forward(self, s_core, s_resnet, s_swin, alphas):
        """
        s_core:   [B, 128]
        s_resnet: [B, 32]
        s_swin:   [B, 16]
        alphas:   [B, 2] in [0, 1]
        """
        fused = s_core.clone()
        a1 = alphas[:, 0:1]
        a2 = alphas[:, 1:2]
        W = self.window

        # Left Boundary: blend Core[0:W] with ResNet tail [32-W:32]
        fused[:, :W] = torch.lerp(s_core[:, :W], s_resnet[:, -W:], a1)

        # Right Boundary: blend Core[128-W:128] with Swin head [0:W]
        fused[:, -W:] = torch.lerp(s_core[:, -W:], s_swin[:, :W], a2)

        return fused  # Dimension strictly preserved at 128!
```

**Pros & Trade-offs**:
- **Zero Dimension Growth**: Keeps the classifier head compact at 128 dimensions.
- **Surgical Perturbation**: Leaves 98%+ of the core representation intact.
- **Limitation**: Inherently 1D (left vs right). Adding a 3rd or 4th backbone requires wrapping into higher dimensions or arbitrary internal partitioning.

---

### 6. Strategy 4: Expanded Embedding with Modular Backbone Mixing Spans

#### 6.1 Conceptual Blueprint
Instead of constraining the fused feature to 128 dimensions, we expand the embedding representation to accommodate dedicated, full-capacity mixing spans for each backbone.

Two layout configurations are supported:

**Option 4A: Sequential Backbone Spans (Append Mode)**
```text
┌──────────────────────────────┬──────────────────────────────┬──────────────────────────────┬─────┐
│      Core Invariant Hub      │      ResNet Mixed Span       │       Swin Mixed Span        │ ... │
│       Channels 0..127        │      Channels 128..159       │      Channels 160..175       │ X/Y │
│       (128 dimensions)       │  LERP(Trunk, ResNet, α₁)     │    LERP(Trunk, Swin, α₂)     │ ... │
└──────────────────────────────┴──────────────────────────────┴──────────────────────────────┴─────┘
```

**Option 4B: Bilateral Far-Left / Far-Right Perimeter Layout**
```text
┌──────────────────────────────┬──────────────────────────────┬──────────────────────────────┐
│      ResNet Mixing Span      │      Core Invariant Hub      │       Swin Mixing Span       │
│       (Far Left: 0..31)      │     (Center: 32..159)        │    (Far Right: 160..175)     │
│   LERP(Trunk_L, ResNet, α₁)  │       (128 dimensions)       │   LERP(Trunk_R, Swin, α₂)    │
└──────────────────────────────┴──────────────────────────────┴──────────────────────────────┘
```

#### 6.2 Mathematical Formulation
Let the student trunk feature provide candidate representations for each domain. For each backbone $k$ with specialist dimension $D_k$:
$$s_{mix}^{(k)} = \text{lerp}(s_{trunk}^{(k)}, s_{sat}^{(k)}, \alpha_k) = (1 - \alpha_k) \cdot s_{trunk}^{(k)} + \alpha_k \cdot s_{sat}^{(k)}$$

The final expanded representation concatenates the untouched core with all modular mixing spans:
$$s_{expanded} = \left[ s_{core}, \; s_{mix}^{(1)}, \; s_{mix}^{(2)}, \; \dots, \; s_{mix}^{(K)} \right] \in \mathbb{R}^{D_{core} + \sum_{k=1}^K D_k}$$

#### 6.3 PyTorch Implementation
```python
class ExpandedBlockLerpFusion(nn.Module):
    """
    Expands the embedding beyond 128 dimensions by allocating dedicated
    modular LERP mixing blocks for each backbone.
    """
    def __init__(self, in_channels=576, core_dim=128, sat_dims=[32, 16], layout="sequential"):
        super().__init__()
        self.layout = layout
        self.core_dim = core_dim
        self.sat_dims = sat_dims

        # Trunk projections providing base representations for each satellite slot
        self.trunk_to_sat_bases = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(inplace=True)
            ) for dim in sat_dims
        ])

    def forward(self, s_core, sat_list, s_trunk_pooled, alphas):
        """
        s_core:          [B, core_dim]
        sat_list:        list of K tensors [B, sat_dim_k]
        s_trunk_pooled:  [B, in_channels] (e.g. 576-dim global pooled student trunk)
        alphas:          [B, K] gating weights
        """
        mixed_blocks = []
        for k, (sat, trunk_proj) in enumerate(zip(sat_list, self.trunk_to_sat_bases)):
            trunk_base = trunk_proj(s_trunk_pooled)
            a_k = alphas[:, k:k+1]
            # Blend trunk context with specialized satellite via LERP
            blended = torch.lerp(trunk_base, sat, a_k)
            mixed_blocks.append(blended)

        if self.layout == "sequential":
            # [Core (128), Sat1 (32), Sat2 (16), ...] -> 176 dims
            return torch.cat([s_core] + mixed_blocks, dim=-1)
        elif self.layout == "bilateral":
            # [Sat1 Far-Left (32), Core Center (128), Sat2 Far-Right (16)] -> 176 dims
            return torch.cat([mixed_blocks[0], s_core, mixed_blocks[1]], dim=-1)
```

**Pros & Trade-offs**:
- **Full Capacity**: Each backbone has dedicated representational bandwidth without crowding or compressing into 128 dimensions.
- **Infinitely Modular**: Any number of heterogeneous backbones (ConvNeXt, DINOv2, CLIP, ViT-H) can be plugged in by adding another mixing block.
- **Trade-off**: Increases classifier input dimension from 128 to 176+ (small parameter overhead, ~9.6k weights for 200 classes).

---

### 7. Experimental Ablation Matrix

When evaluating feature fusion on Tiny-ImageNet-200, we systematically benchmark:

| Method ID | Fusion Formulation | Output Dim | Extra Params | K-Backbone Scaling | Anti-Soup Mechanism |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **M1: Baseline** | Gated Concatenation (`cat[core, a1*sat1, a2*sat2]`) | 176 | 0 | Linear Dim Growth | None (Naive heuristic) |
| **M2: Slotted Radial** | Partitioned Core (`[core_base, lerp(slot1), lerp(slot2)]`) | **128** | 0 | Symmetric Slots | Orthogonal Subspace Slicing |
| **M3a: Residual Bridge** | Unconstrained Residual (`core + sum(ak * (proj(sat) - core))`) | **128** | ~6k | Infinite (Zero Growth) | None (Susceptible if $K \gg 2$) |
| **M3b: Convex Bridge** | Softmax Hull (`alpha0*core + sum(ak*proj(sat))`) | **128** | ~6k | Infinite (Zero Growth) | **Convex Bounded Variance** |
| **M3c: Sparse Top-K** | Mixture-of-Teachers (`Top-M Softmax Gating`) | **128** | ~6k | Infinite (Zero Growth) | **Selective Image Routing (Zero Bleed)** |
| **M4: Boundary Point** | Extrema Interpolation (`lerp(core[0], sat1[-1])`) | **128** | 0 | 1D Boundary ($K=2$) | Flank Point Interface |
| **M5: Expanded Sequential**| Dedicated Mixing Blocks (`[core, lerp(trunk1, sat1), ...]`) | 176+ | ~30k | Infinite (Modular Block) | Dedicated Channel Bandwidth |
| **M6: Expanded Bilateral** | Far-Left / Far-Right Layout (`[lerp_L, core_center, lerp_R]`) | 176 | ~30k | 1D Perimeter ($K=2$) | Polarized Inductive Biases |


