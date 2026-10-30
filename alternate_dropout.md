# Alternate Dropout Strategy: Graph-Guided Cumulative Hub Masking (Curriculum MFH)

## Core Concept
Instead of selecting the cumulative mask order blindly at random, **Graph-Guided Cumulative Hub Masking** determines the deterministic node masking sequence directly from the precomputed filter homophily topology (`filter_graph_topology.pt`) at the start of Stage 2 (balanced across both `ConvNeXt` [1024d] and `DINOv2` [768d]).

---

## 1. Cumulative Node Ordering via Cross-Backbone Homophily Degree
For each filter node $i \in \{0, \dots, 1791\}$, we compute its **Cross-Backbone Neighbor Support Score** from the normalized adjacency matrix $\hat{A}$:
$$s_i^{\text{cross}} = \sum_{j \in \text{Other Teacher}} \hat{A}_{ij}$$

1. **First to be Masked ($0\% \to X\%$ from `start_epoch` to `end_epoch`)**:
   * Filters with the **strongest cross-backbone neighbor connections** ($s_i^{\text{cross}}$ highest) are added to the cumulative mask set $\mathcal{M}_t$ first.
   * Because these filters already have strong homophily bridges to the opposite teacher's filters in $\hat{A}$, they are the easiest to reconstruct via LightGCN diffusion (`H_masked @ A_hat`), creating a smooth easy-to-hard curriculum.
2. **Never Masked (The Top $(100 - X)\%$ Permanent Semantic Hubs $\mathcal{H}$)**:
   * The remaining $(100 - X)\%$ core anchor filters are never masked at any point in training.
   * They serve as permanent, stable **Semantic Hubs** that continuously supply visual activations into the LightGCN propagation steps without "moving-target whiplash."
3. **Final Lock & Adjustment Phase (`end_epoch` $\to$ `total_epochs`)**:
   * Once $\mathcal{M}_t$ reaches the target mask ratio $X\%$ at `end_epoch`, the mask set freezes ($\mathcal{M}_t = \mathcal{M}_{\text{end}}$) for the remaining epochs while Cosine Annealing LR cools to `1e-5`, allowing the GNN homophily gates and classifier heads to fine-tune on a fixed hub-to-receiver topology.

---

## 2. Planned Ablation Comparison
| Strategy | Node Selection Order | Cumulative Property ($\mathcal{M}_t \subset \mathcal{M}_{t+1}$) | Purpose in Paper |
| :--- | :--- | :---: | :--- |
| **1. Standard Cumulative MFH** *(Test First)* | Fixed seeded random permutation (balanced per teacher) | Yes | Proves progressive cumulative hub preservation beats static/random dropout |
| **2. Graph-Guided Cumulative MFH** *(This Doc)* | Ranked by cross-backbone homophily degree $s_i^{\text{cross}}$ | Yes | Proves topology-aware curriculum ordering further improves cross-teacher hallucination |

