# Knowledge Distillation Loss Literature & Taxonomy
## For Bifurcated Homophilic-Heterophilic Architectures

---

### 1. Classical & Modern Logit-Level KD

#### 1.1 Vanilla Logit Knowledge Distillation
- **Paper**: *Distilling the Knowledge in a Neural Network* (Hinton et al., NeurIPS 2014 Workshop)
- **Mechanism**: Temperature-scaled Kullback-Leibler (KL) divergence between teacher and student softened class probability distributions:
  `L_KD = KL(Softmax(z_student / T), Softmax(z_teacher / T)) * (T^2)`
- **Role in our Setup**: Applied directly to the student's **final fused logits** against the teacher consensus/ensemble logits. Transfers the overall semantic decision boundary.

#### 1.2 Decoupled Knowledge Distillation (DKD)
- **Paper**: *Decoupled Knowledge Distillation* (Zhao et al., CVPR 2022)
- **Insight**: Classical KD couples target-class confidence and non-target "dark knowledge". DKD reformulates KD into:
  - **Target Class KD (TCKD)**: Focuses on sample difficulty and confidence.
  - **Non-Target Class KD (NCKD)**: Focuses on inter-class relational knowledge.
- **Role in our Setup**: Advanced upgrade for fused logits; prevents teacher overconfidence from suppressing subtle satellite signals.

---

### 2. Feature & Representation KD

#### 2.1 FitNets (Hinton & Bengio Lineage)
- **Paper**: *FitNets: Hints for Thin Deep Nets* (Romero et al., ICLR 2015)
- **Mechanism**: Direct $L_2$ / Mean Squared Error (MSE) between intermediate student features and teacher hints.
- **Limitation in our Setup**: Fails when feature dimensions vary widely (e.g. 569 core dims vs. 18 satellite dims), leading to extreme gradient imbalance.

#### 2.2 Directional & Review KD (Overhaul KD / ReviewKD)
- **Paper**: *A Comprehensive Overhaul of Feature Distillation* (Heo et al., ICCV 2019) / *Distilling Knowledge via Knowledge Review* (Chen et al., CVPR 2021)
- **Mechanism**: Cosine distance and margin-based directional alignment:
  `L_dir = 1 - CosineSimilarity(s, z)`
- **Role in our Setup**: **Crucial for Heterophilic Satellites**. Because cosine similarity is bounded in `[-1, +1]`, an 18-dim satellite and a 4-dim satellite generate scale-invariant, balanced gradients.

#### 2.3 Relational Knowledge Distillation (RKD)
- **Paper**: *Relational Knowledge Distillation* (Park et al., CVPR 2019)
- **Mechanism**: Transfers pairwise distance and angular relations between samples across a batch (`S = Z @ Z.T`) rather than individual feature values.
- **Role in our Setup**: Ideal for the **Global Homophilic Core** where manifold geometry matters more than absolute coordinates.

#### 2.4 Contrastive Representation Distillation (CRD)
- **Paper**: *Contrastive Representation Distillation* (Tian et al., ICLR 2020)
- **Mechanism**: InfoNCE-style contrastive learning maximizing mutual information between student and teacher representations.

---

### 3. Bifurcated Losses (Homophilic Core vs. Heterophilic Satellites)

In our modular student architecture, we divide distillation into three specialized objectives:

| Component | Target Source | Recommended Loss | Key Paper / Rationale |
| :--- | :--- | :--- | :--- |
| **Fused Logits** | Teacher Ensemble Logits | **Vanilla Logit KD (T=4.0)** | Hinton et al. (2014) |
| **Core Trunk** | Global Core (569 dims) | **Normalized MSE** | Romero et al. (2015) |
| **Satellite 1** | BB1 Local (18 dims) | **Cosine Directional** | Heo et al. (2019) |
| **Satellite 2** | BB2 Local (4 dims) | **Cosine Directional** | Heo et al. (2019) |
| **Cross-Module** | Core vs Satellites | **Orthogonality Penalty** | Disentangled Repr. Learning |

#### 3.1 Orthogonality / Disentanglement Regularization
To prevent the satellite modules from lazily copying core features:
`L_ortho = abs(CosineSim(s_core, s_sat1)) + abs(CosineSim(s_core, s_sat2))`

---

### 4. Implementation Phasing Strategy

1. **Phase 1 (Immediate / Baseline)**:
   - Plain Task CrossEntropy + Standard Logit KD on fused embeddings.
   - Non-graph engine: evaluates the raw inductive bias of the modular student architecture.
2. **Phase 2 (Bifurcated Feature Distillation)**:
   - Add Core Normalized MSE + Satellite Cosine Alignment + Orthogonality loss.
3. **Phase 3 (GNN Topological Distillation)**:
   - Introduce GNN message passing over the Homophily Graph before target extraction.
