> **LEGACY SOURCE AUDIT — NOT PART OF THE ACTIONSHAP REVISION 4 EXPERIMENT.**
> The canonical recommendation paper uses the profile and ItemKNN models,
> MovieLens-1M, and Amazon Digital Music as defined in
> `../../ActionShap_Recommendation_Spec.md` and `../configs/final.yaml`.
> Amazon-Book and DyHuCoG statements below are retained only to document the
> superseded proposal and must not be copied into the current manuscript.

# DyHuCoG — Implementation Specification

**Extracted from:** M. Louhichi, R. Nesmaoui, M. Lazaar, *"DyHuCoG: A Dynamic Hypergraph Cooperative Game for Preference-aware Recommendation"*, International Journal of Intelligent Engineering and Systems, Vol. 19, No. 2, 2026. DOI: 10.22266/ijies2026.0228.54. Received October 8, 2025; revised December 30, 2025. Journal pages 887–902 (16 pages).

**Source PDF:** `/Users/mlouhichi/Desktop/personal/phd/DyHuCoG A Dynamic Hypergraph Cooperative Game for Preference-aware Recommendation.pdf`

**Artifact availability:** The paper contains **no code URL, no data availability statement, and no supplementary material**. The only appendix (Appendix A) covers statistical testing methodology exclusively and contains no model details.

**Extraction scope:** The complete paper was read, including the abstract, Sections 1–5, all nine tables, the Notation List, Appendix A, and the reference list. Figures 1–6 are raster images; their embedded text was not machine-extractable and is not represented here.

---

> # ⚠️ CRITICAL WARNING — READ BEFORE CODING
>
> ## This paper cannot be reimplemented faithfully from its own contents.
>
> The paper presents 24 numbered equations, but the majority of the quantities those equations depend on are never given — neither in the body text, nor in any of the nine tables, nor in the Notation List. This is not a matter of a few missing minor details; the gaps include the core architectural definition and every optimization hyperparameter that controls convergence.
>
> ### Most consequential missing items (all confirmed absent from the entire paper)
>
> | # | Missing item | Why it blocks reimplementation |
> |---|---|---|
> | 1 | **Hypergraph construction** | The single largest gap. There is no incidence matrix $H$, no rule mapping interactions to hyperedges, and no definition of $A$ or $D$ beyond one sentence. The entire construction is described in 2 sentences (Section 3.5). Without this, the model's defining structure is a guess. |
> | 2 | **Embedding dimension $d$** | Symbol is defined in the Notation List and used in Eq. (24); **no numeric value appears anywhere.** |
> | 3 | **Number of propagation layers $L$** | Symbol defined ("L Number of GNN layers"); **no numeric value appears anywhere.** |
> | 4 | **Learning rate $\eta$** | Symbol defined in Notation List; **no numeric value appears anywhere.** |
> | 5 | **The four loss weights $\lambda_{div}$, $\lambda_{ctx}$, $\lambda_{reg}$** | Eq. (13) is the entire training objective and is **unusable as written** without these. Notation List says only "$\lambda$ — Weights for multi-objective loss components". |
> | 6 | **$\lambda_c$ (context weight in Eq. 12)** | Controls the context contribution to the final score. **No value given.** |
> | 7 | **Dropout rate $p$** | "dropout $p$ on embeddings" is stated; **the value is not.** |
> | 8 | **Layer-fusion coefficients $\alpha_l$ (Eq. 9)** | Not given, and it is never stated whether they are uniform $1/(L{+}1)$, tuned, or learned. |
> | 9 | **$sim(i,j)$ — item–item similarity** | **Never defined.** Required by Eq. (16) (training loss), Eq. (23) (the headline diversity metric), and $Diversity(S)$ in Eq. (1). Only hint: "latent semantic space". |
> | 10 | **$sim(u,i)$ — user–item similarity** | **Never defined.** This is the basis of the "preference-aware" mechanism in Eq. (2) — i.e. the paper's central named contribution. |
>
> ### Equations that are mathematically broken or mutually contradictory as printed
>
> - **Eq. (11) is dimensionally broken.** Printed as $y_{ui} = (1 + a_{ui}) \cdot e_i^{\top}$: the right-hand side is a transposed *vector*, $e_u$ does not appear at all, and $y_{ui}$ is never referenced again anywhere in the paper.
> - **Eq. (12) is dimensionally broken.** It adds $g(c_{u,i})$ — which Eq. (17) treats as a *vector*, via $\|g(c_{u,i}) - e_{c_{u,i}}\|_2^2$ — to the *scalar* $\langle e_u, e_i \rangle$. Undefined operation.
> - **Eqs. (11) and (12) mutually contradict each other.** Eq. (11) applies the attention gate $a_{ui}$ **multiplicatively**; Eq. (12) applies it **additively**. The paper gives no rule for which is used at training or inference time.
> - **Eqs. (6) and (7) give two different propagation rules.** Eq. (6) is symmetric-normalized, parameter-free propagation. Eq. (7) has a trainable $W^{(l)}$, a self-term, and asymmetric Shapley-derived normalization. **The paper never states that (7) supersedes (6)**, and both are presented as the model's update rule.
>
> ### The data protocol is self-contradictory
>
> - **"per-user temporal holdout split (70% train, 10% val, 20% test per user; leave-one-out)"** — a 70/10/20 ratio split and leave-one-out are **two mutually exclusive protocols**, stated in a single clause.
> - **Section 4.3 claims cross-validation ensured "no user or item appeared in both the training and test sets"** — impossible for transductive collaborative filtering with a per-user holdout, and directly contradicts the Section 3.8 protocol.
> - **Table 2 reports raw, unfiltered MovieLens-1M statistics** (1,000,209 interactions; 3,706 items) despite the stated `rating > 3` binarization and 5-core filter. Applying `rating > 3` yields 575,281 positives. **There is therefore no post-preprocessing target to validate a pipeline against.**
> - **The Amazon-Book statistics are exactly the canonical LightGCN/HCCF preprocessed split** (52,643 / 91,599 / 2,984,108), which ships as a fixed 80/20 random split with **no ratings and no timestamps** — making both `rating > 3` and "per-user temporal holdout" inapplicable to it.
>
> ### How to use this document
>
> Sections 1–6 are extraction: they contain only what the paper says, with every gap marked **NOT STATED**. Section 8 contains recommended defaults for every unstated value, each flagged as an **implementer assumption**. Never promote a Section 8 value into a claim about the paper. Treat the paper's reported numbers (e.g. NDCG@20 = 0.2775 on MovieLens-1M) as aspirational rather than reproducible targets, and treat the *architecture* — Shapley-weighted hypergraph message passing plus an independent interaction-level attention gate — as the reproducible contribution.

---

## Table of Contents

- [0. Conventions used in this document](#0-conventions-used-in-this-document)
- [1. Model Architecture](#1-model-architecture)
  - [1.1 Problem formulation and notation](#11-problem-formulation-and-notation-section-31)
  - [1.2 Hypergraph construction](#12-hypergraph-construction-section-35)
  - [1.3 Layer-by-layer architecture](#13-layer-by-layer-architecture)
  - [1.4 Forward-pass equations — Eqs. (6)–(12)](#14-forward-pass-equations--eqs-612)
  - [1.5 Users, items and contexts as heterogeneous players](#15-users-items-and-contexts-as-heterogeneous-players)
- [2. The Cooperative Game Component](#2-the-cooperative-game-component)
  - [2.1 Characteristic function — Eqs. (1)–(2)](#21-characteristic-function--eqs-12-section-33)
  - [2.2 Mixing weights α, β, γ, λ_pref](#22-mixing-weights-α-β-γ-λ_pref--stated)
  - [2.3 Shapley value definition and estimation — Eqs. (3)–(5)](#23-shapley-value-definition-and-estimation--eqs-35-section-34)
  - [2.4 How the Shapley weighting enters the model](#24-how-the-shapley-weighting-enters-the-model)
- [3. The Attention Mechanism](#3-the-attention-mechanism)
  - [3.1 Formulation and position in the architecture](#31-formulation-and-position-in-the-architecture)
  - [3.2 Exposing Shapley and attention as separate per-player importance vectors](#32-exposing-shapley-and-attention-as-separate-per-player-importance-vectors)
- [4. Training](#4-training)
  - [4.1 Loss functions — Eqs. (13)–(18)](#41-loss-functions--eqs-1318-section-36)
  - [4.2 Negative sampling](#42-negative-sampling)
  - [4.3 Complete hyperparameter table](#43-complete-hyperparameter-table)
- [5. Data and Evaluation](#5-data-and-evaluation)
  - [5.1 Datasets and preprocessing](#51-datasets-and-preprocessing)
  - [5.2 Dataset statistics — Table 2](#52-dataset-statistics--table-2)
  - [5.3 Evaluation metrics — Eqs. (19)–(23)](#53-evaluation-metrics--eqs-1923-section-37)
  - [5.4 Reported results for fidelity checking](#54-reported-results-for-fidelity-checking)
  - [5.5 Hardware, runtime and complexity](#55-hardware-runtime-and-complexity)
- [6. Complete Catalogue of Ambiguities and Gaps (63 items)](#6-complete-catalogue-of-ambiguities-and-gaps-63-items)
  - [6.1 Blocking gaps (items 1–22)](#61-blocking-gaps--cannot-write-code-without-guessing-items-122)
  - [6.2 Equations broken as printed (items 23–30)](#62-equations-mathematically-broken-as-printed-items-2330)
  - [6.3 Protocol contradictions (items 31–44)](#63-protocol-contradictions-items-3144)
  - [6.4 Notation inconsistencies (items 45–51)](#64-notation-inconsistencies-between-body-and-notation-list-items-4551)
  - [6.5 Total silences (items 52–63)](#65-things-the-paper-is-silent-on-entirely-items-5263)
- [7. Full Equation Index](#7-full-equation-index-all-24)
- [8. Recommended Defaults — IMPLEMENTER ASSUMPTIONS, NOT PAPER CONTENT](#8-recommended-defaults--implementer-assumptions-not-paper-content)
- [9. Verbatim Notation List from the Paper](#9-verbatim-notation-list-from-the-paper)

---

## 0. Conventions used in this document

| Marker | Meaning |
|---|---|
| **NOT STATED** | The paper does not contain this information anywhere — body, tables, Notation List, or Appendix A. It must never be treated as paper content. |
| *verbatim* / quoted blocks | Text reproduced exactly from the paper. |
| **ASSUMPTION** | An implementer decision introduced by this document, not the paper. Confined to Section 8 and explicitly labelled wherever it appears earlier. |
| §x.y | Section of the paper (not of this document) unless stated otherwise. |

Every equation retains the paper's original numbering, Eq. (1) through Eq. (24).

---

## 1. Model Architecture

### 1.1 Problem formulation and notation (Section 3.1)

Verbatim definitions from Section 3.1:

- $\mathcal{U} = \{u_1, \dots, u_{|\mathcal{U}|}\}$ — the set of users.
- $\mathcal{I} = \{i_1, \dots, i_{|\mathcal{I}|}\}$ — the set of items.
- $\mathcal{C}$ — the set of possible contexts. Described as "e.g., time, location, or session" in §3.1; as "genre" in §3.2 and §3.5; as "time, session, genre" in the Notation List.
- Observed interactions are represented as a **bipartite graph** $\mathcal{G} = (\mathcal{U} \cup \mathcal{I}, \mathcal{E})$, where $(u,i) \in \mathcal{E}$ if user $u$ has interacted with item $i$.
- Each interaction may be associated with a context vector $c_{u,i} \in \mathcal{C}$.
- Objective, verbatim: *"The objective is to learn a scoring function $f(u, i, c_{u,i})$ that ranks items for each user, maximizing both accuracy (as measured by NDCG@20, Recall@20, and diversity)."*

**Inconsistency already present at this point:** §3.1 defines $\mathcal{G}$ as a **bipartite graph** over $\mathcal{U} \cup \mathcal{I}$ (context excluded from the node set), whereas the Notation List defines $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ as the "User-item-context **hypergraph**" and $\mathcal{E}$ as the "Set of observed interactions (**hyperedges**) in $\mathcal{G}$". See catalogue item 35.

### 1.2 Hypergraph construction (Section 3.5)

**This is the single largest gap in the paper.** The entire construction is given in two sentences:

> *"We represent user-item-context relations as a hypergraph whose sparse adjacency $A$ and degree $D$ capture higher order connectivity[14]. Standard symmetric normalization yields stable propagation."*

Reference [14] is HCCF (Xia et al., SIGIR 2022).

#### What can be established from the paper

**Nodes.** Users, items, and contexts. From §3.2: *"the process begins with the input layer, where users, items, and context (e.g., genre) are encoded as nodes."* Total node count is therefore $|\mathcal{U}| + |\mathcal{I}| + |\mathcal{C}|$.

**Hyperedges.** The Notation List states that $\mathcal{E}$ is the *"Set of observed interactions (hyperedges) in $\mathcal{G}$."* The only self-consistent reading is **one hyperedge per observed interaction**, incident on the vertex set $\{u,\; i,\; c_{u,i}\}$ — i.e. a 3-uniform hypergraph, or higher-arity if $c_{u,i}$ is multi-valued (as MovieLens genres are, a film having several genres).

**Corroboration from the ablation.** §4.4 defines:

> *"w/o Hypergraph: The hypergraph is collapsed into a simple bipartite graph, removing all high-order user–item–context interactions."*

This confirms that the higher-order structure **is** the inclusion of context nodes in the incidence relation — not a learned or clustered hyperedge set. Collapsing to bipartite = dropping the context node from each hyperedge.

#### What is NOT STATED about the construction

- **NOT STATED:** The incidence matrix $H \in \mathbb{R}^{|V| \times |\mathcal{E}|}$ is never written down anywhere in the paper.
- **NOT STATED:** The relationship between $H$ and $A$. Whether $A$ is the clique-expansion adjacency $H W_e H^{\top} - D_v$, the star/bipartite expansion $\begin{bmatrix} 0 & H \\ H^{\top} & 0\end{bmatrix}$, or some third construction.
- **NOT STATED:** Whether $D$ is the vertex degree matrix $D_v$, the hyperedge degree matrix $D_e$, or both. The Notation List is deliberately vague: "A, D — Hypergraph adjacency and (hyper)degree matrices".
- **NOT STATED:** Whether hyperedges are ever *learned* or *clustered*. The citation to [14] (HCCF) suggests a learned-hyperedge design — HCCF uses a fixed number of learnable hyperedges per node type, with a low-rank parameterized incidence matrix. But the paper's text describes a *fixed sparse $A$ built from observed interactions*, which is structurally incompatible with HCCF's construction. **This is a genuine architectural fork that an implementer must resolve by guessing.** See catalogue item 1.
- **NOT STATED:** Whether self-loops are added.
- **NOT STATED:** Whether hyperedge weights $W_e$ exist as a separate learnable diagonal (standard in HGNN formulations), or whether the Shapley weights of Eq. (8) play that role.
- **NOT STATED:** Any hyperedge count, if hyperedges are not one-per-interaction.

### 1.3 Layer-by-layer architecture

The forward pass, to the extent the paper specifies it:

1. **Embedding layer.** Free (id-based) embedding tables for users, items, and contexts.
   - Dimension $d$: **NOT STATED.** The symbol is defined in the Notation List ("$d$ embedding dim") and used in the complexity analysis of Eq. (24), but **no numeric value appears anywhere in the paper.**
   - Initialization scheme and scale: **NOT STATED.**
2. **Hypergraph propagation**, $L$ layers, per Eq. (7) (or Eq. 6 — the paper gives both; see catalogue item 26).
   - $L$: **NOT STATED.** Notation List defines "L Number of GNN layers"; no value given.
3. **Layer-wise fusion** across $l = 0 \dots L$, per Eq. (9).
   - Fusion coefficients $\alpha_l$: **NOT STATED.**
4. **Interaction-level attention gate**, per Eqs. (10)–(11).
5. **Context-aware scoring**, per Eq. (12).

**Activation functions.** Written as $\sigma(\cdot)$ in Eqs. (6), (7), and (10). **The paper never says what $\sigma$ is** in any of these three places, and it reuses the identical symbol $\sigma$ for the logistic sigmoid in the BPR loss of Eq. (14). See catalogue items 16 and 50.

**Normalization.** Only "standard symmetric normalization" $D^{-1/2} A D^{-1/2}$ (Eq. 6). **No LayerNorm, BatchNorm, or embedding L2-normalization is mentioned anywhere.**

**Dropout.** §3.6 states "dropout $p$ on embeddings". The value of $p$ is **NOT STATED.** Whether dropout is also applied to edges/messages: **NOT STATED.**

**Bias terms.** **NOT STATED** anywhere — Eqs. (7) and (10) show no bias.

**Residual connections.** Eq. (7) contains a self-term $W^{(l)} e_j^{(l)}$, which functions as a transformed residual. No other skip connections are described.

### 1.4 Forward-pass equations — Eqs. (6)–(12)

All equations transcribed verbatim from the paper, with every symbol defined as the paper defines it.

---

#### Eq. (6) — Baseline symmetric-normalized propagation

$$e^{(l+1)} = \sigma\!\left(D^{-1/2} A D^{-1/2} e^{(l)}\right)$$

Introduced by: *"With $\mathbf{e}^{(l)}$ the stacked node embeddings at layer $l$, we update via"*.

| Symbol | Definition (per paper) |
|---|---|
| $e^{(l)}$ | Stacked node embeddings at layer $l$. (The Notation List calls this object $X^{(\ell)}$ — see catalogue item 45.) |
| $A$ | Hypergraph adjacency matrix (construction **NOT STATED**). |
| $D$ | Hypergraph (hyper)degree matrix (vertex vs. hyperedge degree **NOT STATED**). |
| $\sigma$ | Activation function — **NOT STATED which function.** |

This equation is parameter-free. It contains no trainable weights and no Shapley weighting. It is superseded in intent by Eq. (7), but **the paper never says so.**

---

#### Eq. (7) — Shapley-weighted message passing (the propagation actually intended)

$$e_j^{(l+1)} = \sigma\!\left(W^{(l)} e_j^{(l)} + \sum_{k \in \mathcal{N}(j)} w_{jk} \cdot e_k^{(l)}\right)$$

Introduced by: *"To privilege interactions that contribute more under the cooperative game, we weight messages by normalized Shapley coefficients"*.

| Symbol | Definition (per paper) |
|---|---|
| $e_j^{(l)}$ | Embedding of node $j$ at layer $l$. |
| $W^{(l)}$ | "Trainable weight matrix at layer $\ell$" (Notation List). Shape presumably $\mathbb{R}^{d \times d}$; **NOT STATED.** |
| $\mathcal{N}(j)$ | "Neighborhood of node $v$ in the hypergraph" (Notation List, given for $\mathcal{N}(v)$). Whether this means 1-hop via shared hyperedges, or the hyperedge-incidence neighborhood, is **NOT STATED.** |
| $w_{jk}$ | Normalized Shapley-derived edge weight, defined in Eq. (8). |
| $\sigma$ | Activation — **NOT STATED which function.** |

Note that the neighbor sum is **not** symmetrically normalized here; normalization is entirely carried by $w_{jk}$, which sums to 1 over $\mathcal{N}(j)$ by construction (Eq. 8). Note also that $\mathcal{N}$ is overloaded: it denotes the cooperative-game player set in Eqs. (1) and (3), and a graph neighborhood here. See catalogue item 51.

---

#### Eq. (8) — Shapley-derived neighborhood weights

$$w_{jk} = \frac{\hat{\phi}_{jk}}{\sum_{k' \in \mathcal{N}(j)} \hat{\phi}_{jk'}}$$

Introduced by: *"with neighborhood weights:"*.

| Symbol | Definition (per paper) |
|---|---|
| $\hat{\phi}_{jk}$ | The Monte Carlo Shapley estimate associated with the $(j,k)$ interaction. |
| $w_{jk}$ | Row-normalized weight; the Notation List names this normalized object $\tilde{\phi}_{uv}$: *"Normalized Shapley-derived (hyper)edge weight used in message passing."* |

**Critical unresolved indexing mismatch.** Eqs. (4) and (5) define Shapley estimates for a **single player** $j$, written $\hat\phi_j$. Eq. (8) consumes a **pair-indexed** quantity $\hat\phi_{jk}$. **The paper never bridges these two.** Two plausible readings, neither confirmed:
- $\hat\phi_{jk} = \hat\phi_k$ — the weight of a message is the Shapley value of the sending node.
- $\hat\phi_{jk}$ is a separately estimated *interaction-level* value. The abstract supports this: *"quantifies the marginal utility of each **interaction**"* — but Eqs. (3)–(5) are written for players, supporting the former.

See catalogue item 14.

**Sign/positivity concern (not addressed by the paper):** Shapley values may be negative. If any $\hat\phi_{jk} < 0$, Eq. (8) can produce negative weights or a near-zero denominator. The paper mentions clipping ("clipped the extremes") but gives **no thresholds**, and never states that weights are constrained non-negative.

---

#### Eq. (9) — Layer-wise fusion

$$e_j = \sum_{l=0}^{L} \alpha_l\, e_j^{(l)}$$

Introduced by: *"Layer-wise representations are fused to retain both local and longer-range signals as follows:"*.

| Symbol | Definition (per paper) |
|---|---|
| $e_j$ | Final fused embedding of node $j$, used downstream in Eqs. (10) and (12). |
| $\alpha_l$ | Layer-combination coefficient for layer $l$. **Values NOT STATED.** Whether uniform $1/(L{+}1)$ (LightGCN convention), tuned, or learned is **NOT STATED.** |
| $L$ | Number of layers. **Value NOT STATED.** |

Note the sum starts at $l = 0$, so the raw embedding layer is included, giving $L+1$ terms.

$\alpha_l$ collides notationally with $\alpha$ (the NDCG weight in Eq. 1), with $\alpha_{u,i}$ (attention weight in the Notation List), and with $\alpha$ (significance level in Appendix A). See catalogue item 50.

---

#### Eq. (10) — Interaction-level attention gate

$$a_{ui} = \sigma\!\left(W_a [e_u, e_i, l_i]\right)$$

Introduced by: *"Finally, an interaction-level attention gate refines the scoring by conditioning the user, item, and genre/context embeddings as follows:"*.

| Symbol | Definition (per paper) |
|---|---|
| $[\cdot,\cdot,\cdot]$ | Concatenation. |
| $e_u$ | Fused user embedding (from Eq. 9). |
| $e_i$ | Fused item embedding (from Eq. 9). |
| $l_i$ | **NEVER DEFINED IN THE PAPER.** Not in §3.5, not in the Notation List. From the introducing sentence ("user, item, and genre/context embeddings"), $l_i$ is the item's genre/context embedding — likely a typographical rendering of $g_i$ or $\ell_i$ (label/genre vector). **This is a guess.** See catalogue item 17. |
| $W_a$ | Attention projection matrix. **Shape NOT STATED** — whether $\mathbb{R}^{1 \times 3d}$ (producing a scalar gate) or $\mathbb{R}^{d \times 3d}$ (producing a vector gate). This choice changes Eqs. (11) and (12) entirely. See catalogue item 18. |
| $\sigma$ | Activation — **NOT STATED which function.** Context (a "gate") suggests logistic sigmoid, giving $a_{ui} \in (0,1)$, which is consistent with the $(1 + a_{ui})$ form of Eq. (11). |

**No bias term** is shown. **No multi-head structure.** **No temperature.** **No softmax normalization over any candidate set** — so despite being called "attention", as printed this is a per-interaction sigmoid **gate**, not a normalized attention distribution.

Purpose, per §1 (Introduction): *"An interaction-level attention gate complements this mechanism by amplifying high-contribution signals and dampening noisy signals."*

---

#### Eq. (11) — Intermediate score

$$y_{ui} = (1 + a_{ui}) \cdot e_i^{\top}$$

Introduced by: *"yielding an intermediate score:"*.

**⚠️ THIS EQUATION IS MALFORMED AS PRINTED.** Three defects:
1. The right-hand side is a transposed **vector**, so $y_{ui}$ cannot be a scalar score.
2. $e_u$ **does not appear at all** — the "score" has no dependence on the user.
3. $y_{ui}$ is **never referenced again anywhere in the paper.**

The evident intent is a multiplicative residual gate on the inner product:

$$y_{ui} = (1 + a_{ui}) \cdot \langle e_u, e_i \rangle$$

but **this is a reconstruction by the reader, not the paper's text.** See catalogue item 23.

---

#### Eq. (12) — Final context-aware prediction

$$f(u, i, c_{u,i}) = \langle e_u, e_i \rangle + \lambda_c \cdot g(c_{u,i}) + a_{ui}$$

Introduced by: *"The final context-aware prediction is as follows:"*.

| Symbol | Definition (per paper) |
|---|---|
| $\langle e_u, e_i \rangle$ | Inner product of fused user and item embeddings. |
| $g(\cdot)$ | "Context/genre embedding vector" (Notation List entry for $g$); "Context-embedding function" (Notation List entry for $\psi(\cdot)$). The body uses $g$; the Notation List names the function $\psi$. See catalogue item 48. |
| $\lambda_c$ | "Context-alignment weight/temperature" (Notation List entry for $w_c$). **Value NOT STATED.** |
| $a_{ui}$ | Attention gate from Eq. (10), here entering **additively**. |

**⚠️ THIS EQUATION IS DIMENSIONALLY INCONSISTENT.** Eq. (17) treats $g(c_{u,i})$ as a **vector** (it computes $\|g(c_{u,i}) - e_{c_{u,i}}\|_2^2$), so adding it to the **scalar** $\langle e_u, e_i \rangle$ is undefined. An implementer must choose a reduction; candidates include $\langle e_u, g(c_{u,i})\rangle$, $\langle e_i, g(c_{u,i})\rangle$, $\mathbf{1}^{\top} g(c_{u,i})$, or a learned linear projection to a scalar. **None is indicated by the paper.** See catalogue item 24.

**⚠️ EQS. (11) AND (12) ARE MUTUALLY INCONSISTENT.** Eq. (11) applies $a_{ui}$ multiplicatively; Eq. (12) applies it additively. **The paper gives no rule for which form is used**, at training or at inference. See catalogue item 25.

---

#### Closing statement of Section 3.5 (verbatim)

> *"Together, Eqs. (6)-(12) ensure that who contributes via Shapley, how the signal flows via hypergraph propagation, and which relations matter most via attention are jointly optimized[27]."*

This is the paper's clearest statement of the three-way division of labour, and it is the conceptual basis for the separability described in §3.2 of this document.

### 1.5 Users, items and contexts as heterogeneous players

Verbatim, §3.3: *"We instantiate a dynamic cooperative game $(N, v)$ whose players are the users, items, and contexts."*

Notation List entries:
- "$\mathcal{N}$ — Player set (users, items, contexts) in the cooperative game"
- "$S \subseteq \mathcal{N}$ — Coalition (subset of players)"

So the player set is the **union of all three node types** — the same set as the hypergraph node set, $|\mathcal{N}| = |\mathcal{U}| + |\mathcal{I}| + |\mathcal{C}|$.

**NOT STATED about the player representation:**
- Whether Shapley values are computed over the **global** player set $\mathcal{N}$ (cardinality $\sim 10^5$ for Amazon-Book) or restricted per-minibatch / per-user. §3.4 says "we compute $\hat\phi$ over minibatches", which implies restriction, but the scoping rule is never specified.
- Whether the three player types share a single value scale, or are normalized per type. Since Eq. (8) normalizes within a neighborhood $\mathcal{N}(j)$ that may mix types, cross-type scale comparability matters and is unaddressed.
- How a **node-level** Shapley value $\hat\phi_j$ becomes an **edge-level** weight $\hat\phi_{jk}$ (see Eq. 8 discussion above and catalogue item 14).
- Whether context players are treated symmetrically with users and items in coalition formation, or are always co-present with their interaction.

---

## 2. The Cooperative Game Component

### 2.1 Characteristic function — Eqs. (1)–(2) (Section 3.3)

Framing, verbatim: *"The coalition value quantifies the quality of the recommendation configuration produced using only the entities in $S \subseteq N$. Formally, we measure the multi-objective utility as follows."*

---

#### Eq. (1) — Coalition value

$$v(S) = \alpha \cdot NDCG@20(S) + \beta \cdot Diversity(S) + \gamma \cdot ContextScore(S)$$

Following statement, verbatim: *"The weights $(\alpha, \beta, \gamma)$ govern the trade-off among these objectives and are tuned for validation splits."*

---

#### Eq. (2) — Preference-weighted coalition value

$$v_{\text{pref}}(S) = v(S) + \lambda_{\text{pref}} \cdot \sum_{(u,i) \in S} sim(u,i)$$

Introduced by: *"To bias coalitions toward user-preference consistency, we add a preference-weighted variant that increases value when item choices agree with historical or content-based affinities:"*.

Followed by: *"Eqs. (1) and (2) provide a flexible data-driven objective that can be refreshed as distributions shift, enabling responsiveness to seasonality, novelty, and changes in exposure."*

---

#### Critical gaps in the characteristic function

Every one of the four terms in Eqs. (1)–(2) is underspecified:

| Term | Status |
|---|---|
| $NDCG@20(S)$ | **NOT STATED how a ranking is produced from a coalition $S$.** No statement of over which users it is averaged, against which ground truth, or how a model restricted to $S$ is evaluated. Presumably Eqs. (19)–(20) applied to a restricted model, but this is never said. See catalogue item 13. |
| $Diversity(S)$ | **NEVER DEFINED FOR A COALITION.** Eq. (23) defines a *metric* $Diversity_{ILD}$; the paper never states that Eq. (1) reuses it. See catalogue item 12. |
| $ContextScore(S)$ | **NEVER DEFINED ANYWHERE IN THE PAPER.** Not in §3.3, not in §3.7 (metrics), not in the Notation List, not in Appendix A. One of three terms of the characteristic function is simply absent. See catalogue item 11. |
| $sim(u,i)$ | **NEVER DEFINED.** Only characterized as "historical or content-based affinities". This function is the entire basis of the "preference-aware" mechanism that names the contribution. See catalogue item 10. |
| Summation domain $(u,i) \in S$ | Ambiguous: $S$ is a set of *players* (users, items, contexts), so "$(u,i) \in S$" implicitly means the set of interaction pairs both of whose endpoints lie in $S$. Never stated explicitly. |

### 2.2 Mixing weights α, β, γ, λ_pref — STATED

Verbatim, §3.3:

> *"The coalition utility function is governed by four weighting coefficients $\alpha, \beta, \gamma, \lambda_{\text{pref}}$ that jointly balance ranking accuracy, diversity, contextual alignment, and preference consistency. Unless otherwise stated, $\alpha = 0.60$, $\beta = 0.25$, $\gamma = 0.15$ and $\lambda_{\text{pref}} = 0.20$. These values were selected through a grid search in $[0.1,0.8]$ under the constraint $\alpha + \beta + \gamma = 1$, yielding stable performance with a variance of less than 1.5% in NDCG@20."*

| Symbol | **Value (STATED)** | Role |
|---|---|---|
| $\alpha$ | **0.60** | Weight on $NDCG@20(S)$ in $v(S)$ |
| $\beta$ | **0.25** | Weight on $Diversity(S)$ in $v(S)$ |
| $\gamma$ | **0.15** | Weight on $ContextScore(S)$ in $v(S)$ |
| $\lambda_{\text{pref}}$ | **0.20** | Weight on the preference-similarity sum in $v_{\text{pref}}(S)$ |

Search protocol: grid over $[0.1, 0.8]$, constraint $\alpha + \beta + \gamma = 1$.

- **NOT STATED:** the grid step size.
- **NOT STATED:** whether $\lambda_{\text{pref}}$ was searched in the same grid (it is not covered by the $\alpha+\beta+\gamma=1$ constraint).
- **NOT STATED:** whether these values are shared across both datasets or tuned per dataset.

### 2.3 Shapley value definition and estimation — Eqs. (3)–(5) (Section 3.4)

Opening, verbatim: *"We used Shapley values to fairly allocate credit across players[24]. The exact value for player $j$ aggregates the marginal gain from adding $j$ to every coalition that excludes it:"*

---

#### Eq. (3) — Exact Shapley value

$$\phi_j = \sum_{S \subseteq N \setminus j} \frac{|S|!\,\bigl(|N| - |S| - 1\bigr)!}{|N|!}\left[v(S \cup j) - v(S)\right]$$

| Symbol | Definition (per paper) |
|---|---|
| $\phi_j$ | "Shapley value for player $j$ (exact)" (Notation List). |
| $N$ | Player set (users ∪ items ∪ contexts). |
| $v(\cdot)$ | Coalition value, Eq. (1). |

---

#### Eq. (4) — Monte Carlo estimator

$$\hat{\phi}_j = \frac{1}{M} \sum_{m=1}^{M} \left[v(S_m \cup j) - v(S_m)\right]$$

Introduced by: *"The exact computation is combinatorial; therefore, DyHuCoG employs a Monte Carlo estimator that samples coalitions (equivalently, permutations)[25] and averages the observed marginal gains as follows:"*

| Symbol | Definition (per paper) |
|---|---|
| $\hat\phi_j$ | "Monte Carlo estimate of Shapley value for player $j$" (Notation List). |
| $M$ | "Number of Monte Carlo samples / permutations" (Notation List). |
| $S_m$ | The $m$-th sampled coalition. The Notation List defines a related symbol $\mathcal{S}'$ as "Random subset of $\mathcal{N}$ used for sampling". |

---

#### Eq. (5) — Preference-aware Monte Carlo estimator

$$\hat{\phi}_j^{\,pref} = \frac{1}{M} \sum_{m=1}^{M} \left[v_{pref}(S_m \cup j) - v_{pref}(S_m)\right]$$

Introduced by: *"To emphasize personalized alignment, we evaluate the same construction under a preference-weighted value function:"*

Notation List: "$\hat\phi_j^{pref}$ — Preference-aware Shapley value for player $j$".

**Which estimator feeds Eq. (8) is ambiguous.** Eq. (8) writes $\hat\phi_{jk}$ without the $pref$ superscript (i.e. Eq. 4), but the model is named "preference-aware", $\lambda_{\text{pref}} = 0.20$ is given a default value, and §4.8 refers to "The preference-aware Shapley mechanism" as the operative component. **Assume Eq. (5) is used.** See catalogue item 36.

---

#### Stated operational details for Shapley estimation

Verbatim, §3.4: *"As $M$ grows, these estimators concentrate around their respective values[26]. In practice, we compute $\hat\phi$ over minibatches and periodically refresh the edge weights. To reduce the variance in sparse regimes, we applied light temporal smoothing and clipped the extremes before neighborhood normalization in Eq. (8)."*

| Detail | Value / status |
|---|---|
| $M$ (Monte Carlo samples) | **50** in the main/production configuration (STATED, §3.4, §4.6, Table 7). |
| Refresh period $f$ | **Every 10 batches** (STATED). "requiring ~49 updates per epoch on MovieLens-1M (1,000,209 interactions, batch size 2,048)." |
| Scope of estimation | "over minibatches" (STATED qualitatively; precise scoping **NOT STATED**). |
| Variance reduction | "light temporal smoothing" and "clipped the extremes" (STATED qualitatively). **Smoothing coefficient (e.g. EMA $\rho$) NOT STATED. Clipping thresholds/percentiles NOT STATED.** |
| Convergence claim | "variance decreases as $O(1/M)$; empirically, $M=50$ samples achieve ~99% accuracy (MSE ≈ 1.4×10⁻⁵) on MovieLens-1M". |
| MSE reference | "We measure MSE against a high-sample reference ($M_{ref}=1000$) on a validation subset (200 users); accuracy denotes the fraction of estimates within ±5% of this reference." |
| Overhead | "The combined overhead of Shapley estimation and attention mechanisms is ~78% over baseline hypergraph GNN (2,001s vs 1,125s)". |
| Scaling | "The approach scales linearly: on Amazon-Book (3× larger), overhead remains proportional at 1.77× baseline." |

Deployment guidance, verbatim: *"For deployment, we recommend M=50 for production systems (current implementation), M=25 for latency-critical applications (~60% overhead), or M=100 for offline/research scenarios requiring maximum accuracy (~150% overhead)."*

**Table 7 — Monte Carlo Shapley: Convergence and Computational Trade-offs (MovieLens-1M)**

| M | Est. MSE | Accuracy | Runtime | Overhead (×) vs HPCF | Recommendation |
|---|---|---|---|---|---|
| 10 | 1.4×10⁻⁴ | 95% | ~1,460 s | 1.3× | Minimum viable |
| 25 | 5.6×10⁻⁵ | 98% | ~1,800 s | 1.6× | Lightweight |
| **50** | **1.4×10⁻⁵** | **99%** | **2,001 s** | **1.78×** | **Production** |
| 100 | 3.5×10⁻⁶ | 99.5% | ~2,810 s | 2.5× | High-accuracy |

**Sampling scheme gaps — all NOT STATED:**
- The distribution over $S_m$: uniform over all subsets of $N\setminus j$? Uniform random permutation with prefix coalitions (the standard permutation-sampling scheme)? The paper says "samples coalitions (equivalently, permutations)" without committing.
- Whether antithetic sampling, stratified sampling, or any other variance-reduction scheme beyond the mentioned smoothing/clipping is used.
- Whether the same $M$ permutations are **shared** across all players $j$ (standard practice, giving $O(M|N|)$ value evaluations per refresh) or drawn **independently** per player (giving $O(M|N|)$ but with different statistics and far higher cost per player).
- The coalition size distribution.
- Whether Shapley estimation runs under `torch.no_grad()`. Given the 10-batch refresh and caching, values are almost certainly detached constants during backprop, but **this is never stated.** See catalogue item 53.

### 2.4 How the Shapley weighting enters the model

**The paper is unambiguous on this single point.** The Shapley weighting is:

- **NOT** a loss term.
- **NOT** a reweighting of embeddings.
- **NOT** an attention prior.
- **NOT** a sampling weight.

It is an **edge-weight injection into hypergraph message passing**: normalized Shapley coefficients become the neighbor aggregation weights $w_{jk}$ in Eq. (7), via the normalization of Eq. (8).

Corroborating statements, verbatim:

- **Abstract:** *"A preference-aware Shapley estimator (Monte Carlo) quantifies the marginal utility of each interaction, which is then injected as dynamic hyperedge weights into a lightweight hypergraph neural network with an interaction-level-attention gate."*
- **§1 Introduction:** *"These scores are injected as dynamic hyperedge weights so that the information flow reflects the extent to which each interaction improves the underlying ranking objective."*
- **§3.2:** *"These values serve as dynamic weights that guide the flow of information in the graph."*
- **§3.5:** *"To privilege interactions that contribute more under the cooperative game, we weight messages by normalized Shapley coefficients."*
- **§4.4 ablation:** *"w/o Shapley Value: Removes the cooperative game module using uniform edge weights in the GNN."*

The ablation phrasing ("uniform edge weights") is decisive: removing Shapley replaces $w_{jk}$ with $1/|\mathcal{N}(j)|$, confirming that $w_{jk}$ is the sole injection point.

**Ablation cost of removal:** −4.6% NDCG@20 on MovieLens-1M (0.2775 → 0.2647); −6.1% on Amazon-Book (0.0306 → 0.0287). See Table 5 in §5.4.

---

## 3. The Attention Mechanism

### 3.1 Formulation and position in the architecture

**Formulation.** Eqs. (10)–(11), fully transcribed in §1.4 above:

$$a_{ui} = \sigma\!\left(W_a[e_u, e_i, l_i]\right), \qquad y_{ui} = (1 + a_{ui}) \cdot e_i^{\top}$$

and entering the final score additively as the third term of Eq. (12):

$$f(u, i, c_{u,i}) = \langle e_u, e_i \rangle + \lambda_c \cdot g(c_{u,i}) + a_{ui}$$

**Position in the architecture.** After hypergraph propagation and layer fusion (Eq. 9), at the scoring stage. Supporting statements, verbatim:

- §3.5: *"Finally, an interaction-level attention gate refines the scoring by conditioning the user, item, and genre/context embeddings as follows:"*
- §3.2: *"To refine the predictions, an attention mechanism highlights the most salient user–item–context relationships. Finally, the output layer produces weighted recommendation scores."*
- §1: *"An interaction-level attention gate complements this mechanism by amplifying high-contribution signals and dampening noisy signals."*

So the gate sits **outside** the propagation stack — it never influences message passing. This is architecturally important: it is what makes the attention signal independent of the Shapley signal.

**Notation List entries that the body never uses:**
- "$\alpha_{u,i}$ — Attention weight over user-item (and optional context) relation"
- "$A_{att}$ — Attention matrix/tensor over relations"

So the intended object is a per-interaction scalar $\alpha_{u,i} \equiv a_{ui}$, collected into a matrix/tensor $A_{att}$ over all relations. This supports the scalar-gate reading of $W_a$ ($W_a \in \mathbb{R}^{1\times 3d}$), though it is not conclusive. See catalogue items 18 and 47.

**Ablation.** §4.4: *"w/o Attention: Disables the attention layer and relies solely on Shapley-weighted aggregation."* Cost: −3.5% NDCG@20 on MovieLens-1M (0.2775 → 0.2678) and −3.5% on Amazon-Book (0.0306 → 0.0295, which is actually 3.6%; see catalogue item 40).

**NOT STATED about the attention mechanism:**
- The function $\sigma$ in Eq. (10).
- The shape of $W_a$ (scalar vs. vector gate).
- Presence of a bias term.
- Whether $l_i$ is the context embedding $e_{c_{u,i}}$, a separate genre embedding table, or a raw multi-hot genre vector.
- Any multi-head structure, temperature, or normalization.
- Whether the gate is dropout-regularized.
- Whether Eq. (11) or Eq. (12) governs how $a_{ui}$ combines with the inner product.

### 3.2 Exposing Shapley and attention as separate per-player importance vectors

The architecture **does** structurally support extracting both signals separately from a single trained model, and the two are architecturally disjoint — which is the property required for a meaningful comparison.

| Property | Shapley weighting | Attention gate |
|---|---|---|
| Object | $\hat\phi_j^{pref}$, normalized to $w_{jk}$ | $a_{ui}$ |
| Defining equation | Eqs. (5) → (8) | Eq. (10) |
| Produced by | Cooperative game module, from the utility $v_{pref}$ | Learned MLP on fused embeddings |
| Update mechanism | Recomputed every 10 batches; cached; non-differentiable | Differentiable; updated every step by backprop |
| Driven by | Evaluation-metric utility (NDCG + diversity + context + preference) | The BPR ranking loss, Eq. (14) |
| Consumed at | Neighbor aggregation weights **inside** every propagation layer (Eq. 7) | Scoring, **after** all propagation (Eqs. 11/12) |
| Native granularity | Per player (node), then per edge | Per (user, item) interaction |

Because Shapley values are refreshed from a metric-based utility while attention is learned by gradient descent from a pairwise ranking loss, **the two vectors are genuinely different measurements and will not be trivially collinear.** This is what makes comparing them informative. The paper itself never analyzes their relationship.

#### Implementation guidance for dual extraction — **ASSUMPTION / DESIGN DECISION, not paper content**

The paper does not discuss extraction at all. The following is a design recommendation:

1. **Shapley vector.** Maintain a persistent buffer `shapley_phi` of shape `[num_nodes]` (per-player, from Eq. 5) plus the normalized `edge_weight` tensor of shape `[num_edges]` (from Eq. 8). Both are already required by the forward pass, so exposing them costs nothing. Register them as `nn.Module` buffers so they survive `state_dict()` round-trips and are available at inference from a loaded checkpoint.
2. **Attention vector.** Have the attention module return $a_{ui}$ alongside the score, via an explicit `return_attention=True` path or a forward hook, yielding a `[batch]`-shaped (or `[batch, d]`-shaped if a vector gate is chosen) tensor. To obtain a **per-player** attention importance vector directly comparable in shape to `shapley_phi`, aggregate $a_{ui}$ over each node's incident interactions (mean is the natural choice; sum conflates importance with degree).
3. **Keep the two paths strictly separate.** Do not let attention influence $w_{jk}$, and do not let $\hat\phi$ enter Eq. (10). The paper's architecture already enforces this (the gate sits outside the propagation stack), and preserving it is what keeps the two importance vectors independently interpretable.
4. **Record provenance.** Store the refresh step at which `shapley_phi` was last updated, so that a Shapley vector is never compared against an attention vector from a different training state.

**The paper's only interpretability experiment** (§4.7.2, Fig. 4) is a **SHAP waterfall plot over the four utility components of Eqs. (1)–(2)** — ranking utility, diversity, context alignment, and preference consistency — for a single user–item recommendation score. Verbatim: *"The explained output is the final recommendation utility expressed as a function of ranking utility, diversity, context alignment, and preference consistency."* It is **not** an attribution over players, and **not** a comparison of Shapley against attention.

---

## 4. Training

### 4.1 Loss functions — Eqs. (13)–(18) (Section 3.6)

Opening, verbatim: *"The composite objective balances the ranking accuracy, diversity, context alignment, and regularization as follows:"*

---

#### Eq. (13) — Composite objective

$$\mathcal{L} = \mathcal{L}_{rec} + \lambda_{div}\mathcal{L}_{div} + \lambda_{ctx}\mathcal{L}_{ctx} + \lambda_{reg}\mathcal{L}_{reg}$$

| Symbol | Status |
|---|---|
| $\lambda_{div}$ | **NOT STATED** |
| $\lambda_{ctx}$ | **NOT STATED** |
| $\lambda_{reg}$ | **NOT STATED** |

The Notation List offers only: "$\lambda$ — Weights for multi-objective loss components". **No table in the paper gives these values.** Eq. (13) is therefore **unusable as written.** See catalogue item 5.

---

#### Eq. (14) — BPR ranking loss

$$\mathcal{L}_{rec} = -\sum_{(u,i^+,i^-)\in\mathcal{D}} \log \sigma\!\left(f(u,i^+,c_{u,i^+}) - f(u,i^-,c_{u,i^-})\right)$$

Introduced by: *"For implicit feedback, we adopted the BPR (Bayesian Personalized Ranking) formulation to encourage the pairwise ordering of positive items over negative items:"*

| Symbol | Definition |
|---|---|
| $\mathcal{D}$ | The set of training triples $(u, i^+, i^-)$. Construction rule (i.e. negative sampling) described only qualitatively — see §4.2. |
| $f(\cdot)$ | The scoring function of Eq. (12). |
| $\sigma$ | Here unambiguously the logistic sigmoid. |

**No BCE variant is used anywhere.** The Notation List calls this term $\mathcal{L}_{rank}$ while the body calls it $\mathcal{L}_{rec}$ (catalogue item 49).

Note the sum is **unnormalized** — no $1/|\mathcal{D}|$ or batch mean appears. Taken literally this makes the relative scale of $\mathcal{L}_{rec}$ against the mean-normalized $\mathcal{L}_{div}$ (Eq. 15) and $\mathcal{L}_{ctx}$ (Eq. 17) depend on dataset size, which in turn makes the unstated $\lambda$ values dataset-dependent. See catalogue item 27.

---

#### Eq. (15) — Diversity regularizer

$$\mathcal{L}_{div} = -\frac{1}{|\mathcal{U}|}\sum_{u\in\mathcal{U}} ILD(\mathcal{R}_u)$$

Introduced by: *"To counter redundancy in ranked lists, we include an intra-list diversity regulariser:"*

| Symbol | Definition |
|---|---|
| $\mathcal{R}_u$ | The recommended (top-$K$) list for user $u$. **How $\mathcal{R}_u$ is constructed during training is NOT STATED.** |
| $ILD(\cdot)$ | Eq. (16). |

The negative sign means minimizing $\mathcal{L}_{div}$ **maximizes** ILD, as intended.

**⚠️ THIS TERM IS NON-DIFFERENTIABLE AS WRITTEN.** It depends on the top-$K$ list $\mathcal{R}_u$, whose construction requires a non-differentiable sort/top-$K$ operation. **The paper provides no relaxation, no straight-through estimator, and no surrogate.** An implementer must invent one. See catalogue item 15.

---

#### Eq. (16) — Intra-list diversity (ILD)

$$ILD(\mathcal{R}_u) = \frac{2}{K(K-1)}\sum_{1\le k<l\le K}\left[1 - sim(i_k, i_l)\right]$$

Introduced by: *"with ILD (Intra-List Diversity) computed as:"*

| Symbol | Status |
|---|---|
| $K$ | List length. **NOT STATED whether this $K$ is tied to the evaluation cutoff (20) or is a separate training hyperparameter.** See catalogue item 30. |
| $sim(i_k, i_l)$ | **NEVER DEFINED IN THE PAPER.** See catalogue item 9 and the discussion in §5.3. |

---

#### Eq. (17) — Context alignment loss

$$\mathcal{L}_{ctx} = \frac{1}{|\mathcal{E}|}\sum_{(u,i)\in\mathcal{E}}\left\|g(c_{u,i}) - \mathbf{e}_{c_{u,i}}\right\|_2^2$$

Introduced by: *"Context alignment is promoted by matching the learned context signals to the embedding targets as follows:"*

| Symbol | Status |
|---|---|
| $g(c_{u,i})$ | The learned context signal (vector — note this is what makes Eq. 12 dimensionally broken). |
| $\mathbf{e}_{c_{u,i}}$ | "the embedding targets". **NOT STATED whether this is a separate frozen target table, a feature-derived vector, or the same table $g$ maps into.** If it is the same table, the loss is trivially minimized to zero and carries no signal. See catalogue item 59. |
| $|\mathcal{E}|$ | Number of observed interactions. |

---

#### Eq. (18) — Weight-decay / embedding regularization

$$\mathcal{L}_{reg} = \frac{1}{2}\left(\|E_U\|_F^2 + \|E_I\|_F^2\right)$$

Introduced by: *"Weight decay stabilizes the training as follows:"*

| Symbol | Definition |
|---|---|
| $E_U$ | User embedding matrix. |
| $E_I$ | Item embedding matrix. |
| $\|\cdot\|_F$ | Frobenius norm. |

**Note: context embeddings $E_C$ are excluded** from $\mathcal{L}_{reg}$ as written, without comment. Also, if Adam's `weight_decay` is set **and** $\mathcal{L}_{reg}$ is used, regularization is double-counted; the paper never clarifies which mechanism is active. See catalogue items 28 and the hyperparameter table.

---

### 4.2 Negative sampling

Verbatim, §3.6: *"Negatives are sampled from a popularity-aware distribution with periodic hard-negative refreshes."*

That is the entirety of the paper's description. Consequently:

- **NOT STATED:** the negative-to-positive ratio (1:1? 1:$n$?).
- **NOT STATED:** the popularity exponent (e.g. $p(i) \propto \text{pop}(i)^{0.75}$) or any parameterization of the "popularity-aware distribution".
- **NOT STATED:** the hard-negative mining criterion (top-scored non-interacted items? margin-based? dynamic negative sampling?).
- **NOT STATED:** the hard-negative refresh period.
- **NOT STATED:** whether the negative pool excludes validation/test positives (a leakage-relevant choice).
- **NOT STATED:** whether sampling is with or without replacement.

See catalogue item 20.

### 4.3 Complete hyperparameter table

Every hyperparameter mentioned or implied anywhere in the paper, with its status. **There is no hyperparameter table in the paper**; Tables 1–9 are hardware, dataset statistics, main results, cross-validation, ablation, runtime, Shapley convergence, cold-start/cross-dataset, and t-tests. **None contains a model hyperparameter.**

#### Optimization

| Hyperparameter | Value | Source |
|---|---|---|
| Optimizer | **Adam** | §3.6, §3.8 |
| Learning rate $\eta$ | **NOT STATED** (symbol defined in Notation List only) | — |
| Adam $\beta_1, \beta_2, \epsilon$ | **NOT STATED** | — |
| Adam `weight_decay` | **NOT STATED** | — |
| $\lambda_{reg}$ (Eq. 13) | **NOT STATED** | — |
| LR schedule / warmup | **NOT STATED** | — |
| Gradient clipping | **NOT STATED** | — |
| Mixed precision (AMP) | **NOT STATED** | — |
| Training batch size $B$ | **2,048** | §3.4, §4.6 ("batch size 2,048") |
| Evaluation batch size | **1,024** | §3.8 |
| Number of epochs | **NOT STATED** (only early stopping is described) | — |
| Early stopping metric | **validation NDCG@20** | §3.6, §3.8, §4.1 |
| Early stopping patience | **20** | §3.8 ("early stopping patience = 20") |

#### Architecture

| Hyperparameter | Value | Source |
|---|---|---|
| Embedding dimension $d$ | **NOT STATED** | — |
| Number of GNN layers $L$ | **NOT STATED** | — |
| Layer-fusion coefficients $\alpha_l$ | **NOT STATED** | — |
| Dropout rate $p$ | **NOT STATED** ("dropout $p$ on embeddings", value absent) | — |
| Activation $\sigma$ in Eqs. (6), (7), (10) | **NOT STATED** | — |
| $W^{(l)}$ shape | **NOT STATED** (presumably $d\times d$) | — |
| $W_a$ shape | **NOT STATED** (scalar vs. vector gate) | — |
| Embedding initialization | **NOT STATED** | — |
| Bias terms | **NOT STATED** | — |
| Context vocabulary size $|\mathcal{C}|$ | **NOT STATED** for any dataset | — |

#### Loss weights

| Hyperparameter | Value | Source |
|---|---|---|
| $\lambda_{div}$ | **NOT STATED** | — |
| $\lambda_{ctx}$ | **NOT STATED** | — |
| $\lambda_{reg}$ | **NOT STATED** | — |
| $\lambda_c$ (context weight in Eq. 12) | **NOT STATED** | — |
| $K$ in the training ILD (Eq. 16) | **NOT STATED** | — |

#### Cooperative game

| Hyperparameter | Value | Source |
|---|---|---|
| $\alpha$ | **0.60** | §3.3 |
| $\beta$ | **0.25** | §3.3 |
| $\gamma$ | **0.15** | §3.3 |
| $\lambda_{\text{pref}}$ | **0.20** | §3.3 |
| Grid-search range for $\alpha,\beta,\gamma$ | **[0.1, 0.8]**, constraint $\alpha+\beta+\gamma=1$ | §3.3 |
| Grid step size | **NOT STATED** | — |
| $M$ (Monte Carlo samples) | **50** | §3.4, §4.6, Table 7 |
| $f$ (Shapley refresh period) | **10 batches** | §3.4, §4.6 |
| Shapley updates per epoch (ML-1M) | **~49** | §3.4, §4.6 |
| $M_{ref}$ (MSE reference) | **1,000**, on a validation subset of **200 users** | §4.6 |
| Temporal smoothing coefficient | **NOT STATED** | — |
| Clipping thresholds | **NOT STATED** | — |
| Coalition sampling distribution | **NOT STATED** | — |
| Detach/`no_grad` on $\hat\phi$ | **NOT STATED** | — |

#### Negative sampling

| Hyperparameter | Value | Source |
|---|---|---|
| Distribution family | "popularity-aware" (qualitative only) | §3.6 |
| Popularity exponent | **NOT STATED** | — |
| Negative:positive ratio | **NOT STATED** | — |
| Hard-negative criterion | **NOT STATED** | — |
| Hard-negative refresh period | **NOT STATED** | — |

#### Environment and protocol

| Hyperparameter | Value | Source |
|---|---|---|
| Python | **3.8** | §3.8 |
| PyTorch | **2.0.1** | §3.8 |
| Device | **CUDA** | §3.8 |
| Data-loader workers | **4** | §3.8 |
| Random seeds | **{42, 43, 44, 45, 46}** (5 runs) | §3.8, §4.1 |
| Evaluation cutoffs $K$ | **{5, 10, 20}** stated; **only $K=20$ ever reported** | §3.8 |
| Tuning protocol | "tuned on a held-out validation set using a shared grid across the datasets[28]" — **the grid itself is never reported** | §3.6 |

Because the grid is never reported and no per-dataset values are given, **the paper provides no way to know whether MovieLens-1M and Amazon-Book used the same $d$, $L$, or $\eta$.** See catalogue item 61.

---

## 5. Data and Evaluation

### 5.1 Datasets and preprocessing

**Datasets used.** **MovieLens-1M** — described as "denser, explicit feedback" (§1) and "relatively dense and well-curated" (§4.1). **Amazon-Book** — "sparser, long-tail implicit feedback" (§1) and "sparser and more challenging, providing a robust testbed for diversity-aware recommendations" (§4.1).

A third dataset, **Yelp2018**, appears **only** in Table 8 (cross-dataset column) with **no statistics, no preprocessing description, and no mention in §4.1**. See catalogue item 39.

**Versions.** **NOT STATED** for any dataset. No download URLs. The Amazon-Book review category and snapshot year are **NOT STATED**.

**Shared protocol, verbatim (§3.8):**

> *"Shared protocol across all methods: Adam optimizer; early stopping patience = 20; per-user temporal holdout split (70% train, 10% val, 20% test per user; leave-one-out); implicit positive if rating > 3; minimum 5 interactions; metrics reported at K ∈ {5, 10, 20}; 5 runs with random seeds {42, 43, 44, 45, 46} evaluation batch size = 1024; 4 data-loader workers; device = CUDA."*

Decomposed:

| Preprocessing step | Stated value | Gaps |
|---|---|---|
| Binarization | **implicit positive if rating > 3** (so ratings 4–5 on MovieLens' 1–5 scale) | Whether ratings ≤ 3 are discarded entirely or retained as explicit negatives: **NOT STATED.** |
| $k$-core filtering | **minimum 5 interactions** (5-core) | Whether applied to users only, or iteratively to both users and items: **NOT STATED.** Whether applied before or after the `rating > 3` filter: **NOT STATED** — and this choice materially changes the resulting dataset. |
| Split | **per-user temporal holdout, 70% train / 10% val / 20% test** | Requires timestamps — available for MovieLens-1M, but not for the Amazon-Book preprocessing whose statistics the paper reports. |
| Split, contradictory addendum | "**leave-one-out**" | ⚠️ A 70/10/20 ratio split and leave-one-out are **mutually exclusive protocols**, stated in one clause. See catalogue item 31. |
| Candidate set at evaluation | §3.7: *"To avoid leakage, candidates exclude training positives other than the evaluation target."* | Implies **full-catalog ranking with training positives masked** rather than sampled-negative ranking, but the phrasing is ambiguous. See catalogue item 60. |

**Context features.**
- MovieLens-1M context is **genre** ("context (e.g., genre)" §3.2; "genre/context embeddings" §3.5).
- **NOT STATED:** whether genre is a multi-hot vector, and how multiple genres per film are pooled into a single $c_{u,i}$ and $\mathbf{e}_{c_{u,i}}$.
- **Amazon-Book context is never specified.** No genre field exists in the standard preprocessed Amazon-Book, so what plays the role of context there is entirely unknown. See catalogue item 57.
- Yelp2018 context: **NOT STATED.**
- $|\mathcal{C}|$ for any dataset: **NOT STATED.**
- Whether context embeddings are trained, frozen, or feature-derived: **NOT STATED.**

### 5.2 Dataset statistics — Table 2

**Table 2. Dataset Statistics Used for Model Training and Evaluation**

| Dataset | MovieLens-1M | Amazon-Book |
|---|---|---|
| Users | 6,040 | 52,643 |
| Items | 3,706 | 91,599 |
| Interactions | 1,000,209 | 2,984,108 |
| Density | 0.0447 | 0.0006 |

**Internal arithmetic checks (both pass as stated):**
- MovieLens-1M: $1{,}000{,}209 / (6{,}040 \times 3{,}706) = 1{,}000{,}209 / 22{,}384{,}240 = 0.0447$ ✓
- Amazon-Book: $2{,}984{,}108 / (52{,}643 \times 91{,}599) \approx 2.984\times10^6 / 4.822\times10^9 = 6.19\times10^{-4} \approx 0.0006$ ✓

**⚠️ But these statistics are inconsistent with the stated preprocessing, in two independent ways:**

1. **MovieLens-1M numbers are the raw, unfiltered dataset.** 1,000,209 is the full MovieLens-1M rating count and 3,706 the full item count. Applying the stated `rating > 3` filter yields **575,281** positives, and the 5-core filter would reduce items further. **Table 2 therefore reports pre-filtering statistics and gives an implementer no target to validate a correctly filtered pipeline against.** See catalogue item 33.

2. **The Amazon-Book triple (52,643 / 91,599 / 2,984,108) is exactly the canonical LightGCN/HCCF preprocessed Amazon-Book**, which is distributed as a fixed 80/20 random train/test split containing **no ratings and no timestamps**. Neither `rating > 3` nor "per-user temporal holdout 70/10/20" can be applied to it. See catalogue item 34.

**Yelp2018 statistics: NOT STATED anywhere** (the dataset appears only as a column header in Table 8). See catalogue item 39.

**Cold-start definition (§4.8):** users and items with **≤ 5 training interactions**. Note this interacts oddly with the stated 5-core minimum-5-interactions filter.

**Cross-dataset density range claimed (§4.8):** "densities ranging from 0.0006 to 0.0447 (74× variation)".

### 5.3 Evaluation metrics — Eqs. (19)–(23) (Section 3.7)

Protocol, verbatim: *"We evaluate at a fixed cutoff (e.g., $K = 20$) and report user-averaged scores with mean ± std over five runs. To avoid leakage, candidates exclude training positives other than the evaluation target."*

---

#### Eq. (19) — NDCG@20

$$NDCG@20 = \frac{1}{|\mathcal{U}|}\sum_{u\in\mathcal{U}}\frac{DCG_u@20}{IDCG_u@20}$$

Introduced by: *"NDCG@20. The Normalized Discounted Cumulative Gain (NDCG) measures position-sensitive ranking quality as follows:"*

---

#### Eq. (20) — Per-user DCG

$$DCG_u@20 = \sum_{k=1}^{20}\frac{2^{rel_{u,k}} - 1}{\log_2(k+1)}$$

Introduced by: *"with the per-user discounted gain:"*. Followed by: *"Here $rel_{u,k}$ is binary for implicit feedback; IDCG is computed per user to normalize into [0,1]."*

| Symbol | Definition |
|---|---|
| $rel_{u,k}$ | Relevance of the item at rank $k$ for user $u$; **binary** for implicit feedback. |
| $IDCG_u@20$ | Ideal DCG, computed per user. Exact form **NOT STATED** but standard: $\sum_{k=1}^{\min(20,\,|\text{test}_u|)} 1/\log_2(k+1)$. |

Note that the exponential gain $2^{rel}-1$ is redundant given binary relevance (it equals $rel$), but harmless. See catalogue item 29.

---

#### Eq. (21) — Recall@20

$$Recall@20 = \frac{TP}{TP + FN}$$

Introduced by: *"Recall@20: Proportion of relevant items successfully recommended from all relevant items for a user."*

**NOT STATED:** whether this is computed per user then averaged (macro) or pooled across users (micro). The wording "for a user" plus §3.7's "report user-averaged scores" implies macro-averaging, but Eq. (21) as written is a pooled formula.

---

#### Eq. (22) — Catalog coverage

$$Coverage = \frac{\left|\bigcup_{u\in\mathcal{U}}\mathcal{R}_u\right|}{|\mathcal{I}|}$$

Introduced by: *"Catalog coverage reflects the fraction of items that surface across users:"*. Followed by: *"We also analyzed head/tail coverage by popularity deciles to quantify exposure deconcentration."*

**The head/tail decile analysis is described but its numbers are never reported.**

**NOT STATED:** the list length $K$ used for $\mathcal{R}_u$ in the coverage computation (presumably 20).

---

#### Eq. (23) — Intra-list diversity (reported metric)

$$Diversity_{ILD} = 1 - \frac{1}{|\mathcal{U}|}\sum_{u\in\mathcal{U}}\frac{2}{K(K-1)}\sum_{1\le i<j\le K} sim(i,j)$$

Introduced by: *"Intra-list Diversity complementarity is summarized by:"*. Followed by: *"The Intra-List Diversity (ILD) score therefore measures the average dissimilarity among recommended items across all users, normalized to the range [0,1]. This metric provides a faithful quantitative view of content diversity in the latent semantic space."*

**⚠️ $sim(i,j)$ IS NEVER DEFINED, AND THIS IS THE MOST CONSEQUENTIAL UNDEFINED FUNCTION IN THE PAPER**, because the same symbol appears in three distinct roles:
1. The **training loss** (Eq. 16, via $\mathcal{L}_{div}$).
2. The **reported headline diversity metric** (Eq. 23, the "Diversity" column of Table 3).
3. Implicitly in $Diversity(S)$ within the **characteristic function** (Eq. 1), if Eq. (23) is what Eq. (1) reuses.

The only hint is *"in the latent semantic space"*, which suggests cosine similarity between learned item embeddings. **But that reading creates a serious methodological problem:** it makes the diversity *metric* model-dependent, since each baseline (MF, NCF, LightGCN, RecDCL, HCCF, HPCF) has its own embedding space with its own similarity geometry. Diversity numbers computed that way are not comparable across models, which would invalidate the entire "Diversity" column of Table 3.

The alternative reading — $sim$ computed on fixed content features (e.g. Jaccard or cosine on genre vectors) — keeps the metric model-independent and comparable, and is what a rigorous comparison requires. **The paper does not resolve which is used.** See catalogue item 9.

Note also that Eq. (23) and Eqs. (15)–(16) are algebraically equivalent formulations (Eq. 23 factors the $1-\text{sim}$ out of the pairwise sum), so the reported metric and the training regularizer measure the same quantity — but the paper never states that they share a $sim$ implementation.

---

#### Statistical protocol (§4.1 and Appendix A)

- 5 runs with seeds {42, 43, 44, 45, 46}; results as mean ± standard deviation.
- 95% confidence intervals computed for all metrics.
- Paired **two-sided t-tests** on per-user NDCG@20. Verbatim (Appendix A): *"We report statistical significance using per-user metrics on the MovieLens-1M test set ($n = 6040$ test users; $df = 6039$). For each user, NDCG@20 is computed on the held-out test interactions and then paired across methods."*
- Effect size: paired **Cohen's $d_z$**.
- **Holm–Bonferroni** FWER control at $\alpha = 0.05$ over $k = 6$ pairwise comparisons: *"sort p-values ascending $p_{(1)} \le \dots \le p_{(k)}$, then compare $p_{(i)}$ to $\frac{\alpha}{k - i + 1}$. If $p_{(i)}$ exceeds its threshold, stop and retain the remaining null hypotheses; otherwise continue."*
- Distribution-free robustness check: **one-sided Wilcoxon signed-rank test** on per-user differences (alternative: DyHuCoG > baseline). *"All conclusions remain significant (p < 0.001 for all baselines)."*
- Diagnostics: Fig. 5 (distribution of per-user differences DyHuCoG − HPCF) and Fig. 6 (Q–Q plot of the same). Both are images; no numeric content extractable.

**Metrics are stated to be reported at $K \in \{5, 10, 20\}$, but only $K = 20$ appears anywhere in the paper.** See catalogue item 37.

### 5.4 Reported results for fidelity checking

#### Table 3 — Main Performance Comparison on MovieLens-1M and Amazon-Book

| Dataset | Model | NDCG@20 | Recall@20 | Coverage | Diversity (ILD) |
|---|---|---|---|---|---|
| **MovieLens-1M** | MF | 0.1200 ± 0.0025 | 0.0880 ± 0.0020 | 0.239 ± 0.010 | 0.367 ± 0.009 |
| | NCF | 0.1300 ± 0.0026 | 0.1100 ± 0.0022 | 0.269 ± 0.011 | 0.398 ± 0.008 |
| | LightGCN | 0.2130 ± 0.0030 | 0.1790 ± 0.0027 | 0.308 ± 0.010 | 0.423 ± 0.007 |
| | RecDCL | 0.2296 ± 0.0032 | 0.1918 ± 0.0029 | 0.324 ± 0.010 | 0.445 ± 0.007 |
| | HCCF | 0.2470 ± 0.0035 | 0.2050 ± 0.0031 | 0.327 ± 0.009 | 0.448 ± 0.006 |
| | HPCF | 0.2528 ± 0.0036 | 0.2098 ± 0.0032 | 0.342 ± 0.009 | 0.461 ± 0.006 |
| | **DyHuCoG** | **0.2775 ± 0.0039** | **0.2362 ± 0.0036** | **0.397 ± 0.011** | **0.516 ± 0.005** |
| **Amazon-Book** | MF | 0.0049 ± 0.0005 | 0.0093 ± 0.0009 | 0.168 ± 0.013 | 0.425 ± 0.014 |
| | NCF | 0.0085 ± 0.0008 | 0.0142 ± 0.0014 | 0.193 ± 0.012 | 0.458 ± 0.013 |
| | LightGCN | 0.0236 ± 0.0024 | 0.0320 ± 0.0032 | 0.226 ± 0.011 | 0.491 ± 0.012 |
| | RecDCL | 0.0255 ± 0.0026 | 0.0346 ± 0.0035 | 0.242 ± 0.010 | 0.512 ± 0.011 |
| | HCCF | 0.0258 ± 0.0026 | 0.0344 ± 0.0034 | 0.248 ± 0.010 | 0.520 ± 0.011 |
| | HPCF | 0.0270 ± 0.0027 | 0.0359 ± 0.0036 | 0.259 ± 0.010 | 0.535 ± 0.011 |
| | **DyHuCoG** | **0.0306 ± 0.0031** | **0.0417 ± 0.0042** | **0.336 ± 0.012** | **0.602 ± 0.010** |

**Headline improvements over HPCF (state-of-the-art baseline), verbatim from the abstract and §4.7.1:**

| Dataset | NDCG@20 | Recall@20 |
|---|---|---|
| MovieLens-1M | **+9.77%** | **+12.58%** |
| Amazon-Book | **+13.33%** | **+16.16%** |

Arithmetic verification: $0.2775/0.2528 = 1.0977$ ✓; $0.2362/0.2098 = 1.1258$ ✓; $0.0306/0.0270 = 1.1333$ ✓; $0.0417/0.0359 = 1.1616$ ✓. All four headline claims are consistent with Table 3.

**Coverage and diversity gains over HPCF (§4.5, §5):**

| Dataset | Coverage | ILD |
|---|---|---|
| MovieLens-1M | 0.342 → 0.397 (**+16.1%**) | 0.461 → 0.516 (**+11.9%**) |
| Amazon-Book | 0.259 → 0.336 (**+29.7%**) | 0.535 → 0.602 (**+12.5%**) |

---

#### Table 4 — Cross-Validation Results (NDCG@20, mean ± std across folds)

| Dataset | Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± Std | Min–Max |
|---|---|---|---|---|---|---|---|---|
| **ML-1M** | MF | 0.1175 | 0.1188 | 0.1200 | 0.1212 | 0.1225 | 0.1200 ± 0.0025 | [0.1175, 0.1225] |
| | NCF | 0.1274 | 0.1287 | 0.1300 | 0.1313 | 0.1326 | 0.1300 ± 0.0026 | [0.1274, 0.1326] |
| | LightGCN | 0.2100 | 0.2115 | 0.2130 | 0.2145 | 0.2160 | 0.2130 ± 0.0030 | [0.2100, 0.2160] |
| | RecDCL | 0.2264 | 0.2280 | 0.2296 | 0.2312 | 0.2328 | 0.2296 ± 0.0032 | [0.2264, 0.2328] |
| | HCCF | 0.2435 | 0.2453 | 0.2470 | 0.2487 | 0.2504 | 0.2470 ± 0.0035 | [0.2435, 0.2504] |
| | HPCF | 0.2492 | 0.2510 | 0.2528 | 0.2546 | 0.2564 | 0.2528 ± 0.0036 | [0.2492, 0.2564] |
| | **DyHuCoG** | 0.2736 | 0.2756 | 0.2775 | 0.2794 | 0.2813 | **0.2775 ± 0.0039** | [0.2736, 0.2813] |
| **Amazon-Book** | MF | 0.0044 | 0.0047 | 0.0049 | 0.0051 | 0.0054 | 0.0049 ± 0.0005 | [0.0044, 0.0054] |
| | NCF | 0.0076 | 0.0081 | 0.0085 | 0.0089 | 0.0093 | 0.0085 ± 0.0008 | [0.0076, 0.0093] |
| | LightGCN | 0.0212 | 0.0224 | 0.0236 | 0.0248 | 0.0260 | 0.0236 ± 0.0024 | [0.0212, 0.0260] |
| | RecDCL | 0.02293 | 0.02421 | 0.02548 | 0.02675 | 0.02803 | 0.02548 ± 0.00255 | [0.02293, 0.02803] |
| | HCCF | 0.0232 | 0.0245 | 0.0258 | 0.0271 | 0.0284 | 0.0258 ± 0.0026 | [0.0232, 0.0284] |
| | HPCF | 0.02426 | 0.02561 | 0.02696 | 0.02831 | 0.02966 | 0.02696 ± 0.00270 | [0.02426, 0.02966] |
| | **DyHuCoG** | 0.0275 | 0.0291 | 0.0306 | 0.0321 | 0.0337 | **0.0306 ± 0.0031** | [0.0275, 0.0337] |

**⚠️ Do not use Table 4 as a variance reference.** Every model's five fold values form an **exact arithmetic progression** centered on that model's Table 3 mean — for all 7 models on both datasets. Real cross-validation does not produce this pattern. See catalogue item 44.

Validation strategies claimed (§4.3): *"we employed both five-fold cross-validation and repeated random sub-sampling validation. In each fold, the data were split into training, validation, and test sets, ensuring that no user or item appeared in both the training and test sets."* (This last clause is impossible for transductive CF — see catalogue item 32.)

---

#### Table 5 — Component-Wise Ablation Results (NDCG@20)

| Component | ML-1M NDCG@20 | % Drop | Amazon-Book NDCG@20 | % Drop |
|---|---|---|---|---|
| Full DyHuCoG | 0.2775 | – | 0.0306 | – |
| w/o Shapley Value | 0.2647 | 4.6% | 0.0287 | 6.1% |
| w/o Hypergraph | 0.2586 | 6.8% | 0.0279 | 8.9% |
| w/o Attention | 0.2678 | 3.5% | 0.0295 | 3.5% |
| w/o Context | 0.2547 | 8.2% | 0.0272 | 11.0% |
| w/o Diversity | 0.2614 | 5.8% | 0.0288 | 5.8% |

**Ablation variant definitions, verbatim (§4.4):**
- *"w/o Shapley Value: Removes the cooperative game module using uniform edge weights in the GNN."*
- *"w/o Hypergraph: The hypergraph is collapsed into a simple bipartite graph, removing all high-order user–item–context interactions."*
- *"w/o Attention: Disables the attention layer and relies solely on Shapley-weighted aggregation."*
- *"w/o Context: Excludes context embeddings from the model."*
- *"w/o Diversity: Removes the diversity-promoting term from the loss function."*

Ranked by importance (per §4.4 and §4.7.3): **Context (largest) > Hypergraph > Shapley > Diversity > Attention (smallest)** on MovieLens-1M.

**Arithmetic discrepancies in Table 5** (see catalogue item 40): Amazon-Book w/o Attention $0.0295/0.0306$ is a 3.6% drop, listed as 3.5%; w/o Diversity $0.0288/0.0306$ is a 5.9% drop, listed as 5.8%. §4.7.3 also garbles the hypergraph row: *"Eliminating the hypergraph topology led to a 6.8% performance decline (0.2586 on MovieLens-1M and 0.0279 on Amazon-Book), corresponding to drops of 6.8% and 8.9%, respectively."*

---

#### Table 8 — Cold-start and cross-dataset robustness (NDCG@20)

| Method | Cold-Start User | Cold-Start Item | ML | Amazon | Yelp |
|---|---|---|---|---|---|
| MF | 0.026 | 0.025 | 0.120 | 0.005 | 0.003 |
| LightGCN | 0.047 | 0.044 | 0.213 | 0.024 | 0.014 |
| HCCF | 0.054 | 0.051 | 0.247 | 0.026 | 0.016 |
| HPCF | 0.055 | 0.052 | 0.253 | 0.027 | 0.017 |
| **DyHuCoG** | **0.061** | **0.057** | **0.278** | **0.031** | **0.019** |
| Improvement | +10.9% | +9.6% | +9.9% | +14.8% | +11.8% |

Cold-start definition: users and items with **≤ 5 training interactions** (§4.8).

More precise values from the §4.8 body text: *"On MovieLens-1M, DyHuCoG achieves NDCG@20 of 0.0606 for cold-start users and 0.0571 for cold-start items, representing +9.8% improvements over HPCF (0.0552 and 0.0520)."* These conflict with the table's stated +10.9% / +9.6% (catalogue item 41), and the cross-dataset percentages conflict with Table 3's derived values (catalogue item 42).

Note NCF and RecDCL are absent from Table 8 without explanation.

---

#### Table 9 — Paired t-tests on per-user NDCG@20 (MovieLens-1M), df = 6039

| Comparison | t | df | p-value | Cohen's $d_z$ | Holm $\alpha_i$ | Sig. |
|---|---|---|---|---|---|---|
| DyHuCoG vs HPCF | 46.38 | 6039 | 1.81e-270 | 1.3345 | 0.050000 | ✓ |
| DyHuCoG vs RecDCL | 92.72 | 6039 | < 1e-300 | 2.6677 | 0.008333 | ✓ |
| DyHuCoG vs HCCF | 61.21 | 6039 | < 1e-300 | 1.7610 | 0.010000 | ✓ |
| DyHuCoG vs LightGCN | 132.19 | 6039 | < 1e-300 | 3.8035 | 0.012500 | ✓ |
| DyHuCoG vs NCF | 311.13 | 6039 | < 1e-300 | 8.9518 | 0.016667 | ✓ |
| DyHuCoG vs MF | 341.76 | 6039 | < 1e-300 | 9.8330 | 0.025000 | ✓ |

Overall significance claim (§4.7.1): *"all p < 10⁻⁶, Holm-Bonferroni corrected"*.

**The Holm thresholds are misassigned** (catalogue item 43): under the paper's own stated rule $\alpha/(k-i+1)$, the *smallest* p-value should receive $\alpha/6 = 0.008333$. The table instead assigns 0.050000 to the HPCF comparison, which has the *largest* p-value, and 0.008333 to RecDCL. The conclusions are unaffected given the magnitudes involved, but the table is internally wrong.

### 5.5 Hardware, runtime and complexity

#### Table 1 — Hardware Configuration

| Component | Specification |
|---|---|
| Chip | NVIDIA Ada Lovelace (RTX 4090) |
| CPU | Intel Core i9-14900K (24 cores) |
| GPU | NVIDIA GeForce RTX 4090 24GB |
| CUDA Cores | 16,384 |
| Memory | 48GB |
| Storage | 2TB SSD |

Software: Python **3.8**, PyTorch **2.0.1** (§3.8).

---

#### Table 6 — Runtime and Scalability Analysis

Units: Training Time in seconds; Inference in ms per user-item query; Memory in GB (peak).

| Dataset | Model | Training Time | Inference | Memory | Scale Factor (× MF) |
|---|---|---|---|---|---|
| **MovieLens-1M** | MF | 485.2 ± 19.3 | 0.54 ± 0.02 | 2.1 ± 0.1 | 1.00 ± 0.00 |
| | NCF | 675.4 ± 23.1 | 0.76 ± 0.03 | 2.7 ± 0.1 | 1.41 ± 0.08 |
| | LightGCN | 785.6 ± 25.1 | 0.88 ± 0.04 | 3.2 ± 0.2 | 1.63 ± 0.10 |
| | RecDCL | 968.4 ± 29.8 | 1.02 ± 0.05 | 3.6 ± 0.2 | 1.89 ± 0.12 |
| | HCCF | 952.3 ± 28.4 | 1.06 ± 0.05 | 3.8 ± 0.2 | 1.96 ± 0.12 |
| | HPCF | 1124.6 ± 31.8 | 1.18 ± 0.06 | 4.1 ± 0.2 | 2.19 ± 0.14 |
| | **DyHuCoG** | **2000.2 ± 41.2** | **1.84 ± 0.08** | **4.4 ± 0.3** | **3.41 ± 0.19** |
| **Amazon-Book** | MF | 2201.7 ± 69.3 | 2.46 ± 0.12 | 8.5 ± 0.4 | 1.00 ± 0.00 |
| | NCF | 3072.9 ± 82.6 | 3.42 ± 0.17 | 11.1 ± 0.5 | 1.39 ± 0.10 |
| | LightGCN | 3645.2 ± 88.2 | 4.05 ± 0.20 | 12.8 ± 0.6 | 1.65 ± 0.11 |
| | RecDCL | 4526.8 ± 104.6 | 5.04 ± 0.25 | 15.6 ± 0.8 | 2.05 ± 0.14 |
| | HCCF | 4418.5 ± 101.2 | 4.92 ± 0.25 | 15.3 ± 0.8 | 2.00 ± 0.14 |
| | HPCF | 5234.2 ± 118.6 | 5.48 ± 0.27 | 16.8 ± 0.8 | 2.23 ± 0.15 |
| | **DyHuCoG** | **9278.9 ± 140.1** | **8.52 ± 0.43** | **17.9 ± 0.9** | **3.46 ± 0.24** |

Stated ratios: DyHuCoG trains in ≈**1.78×** HPCF's time on MovieLens-1M (2000.2 s vs 1124.6 s) and a "similar" ratio on Amazon-Book (9278.9 s vs 5234.2 s, which is 1.77×). Memory: 4.4 vs 4.1 GB (ML-1M) and 17.9 vs 16.8 GB (Amazon-Book), *"primarily because of storing Shapley estimates and attention weights."*

---

#### Eq. (24) — Training complexity per epoch

$$O\bigl((L + 1)\,m\,d\bigr) + O\bigl((M/f)\,m\bigr)$$

Introduced by: *"Let $n_u, n_i$ be users/items, $m$ interactions, $d$ embedding dim, $L$ layers, $M$ Shapley samples, and $f$ the Shapley refresh period (batches). One training epoch for DyHuCoG costs"*.

Followed by: *"where the first term is sparse hypergraph propagation + ranking and the second is amortized preference-aware Shapley. HCCF is $O((L+1)md)$. Inference: per user-item score $O(d)$; top-$K$ per user $O(n_i d)$ (scan) or $O(d\log n_i)$ with ANN. Memory: $O((n_u + n_i)d + m)$ plus cached attention/Shapley weights ($\approx 1.4 \times$ HCCF empirically). Streaming/online updates: for $\Delta m$ new events, localized refresh is $O(L\Delta m d) + O((M/f)\Delta m)$; increasing $f$ or reducing $M$ trades a small accuracy drop for lower update cost."*

| Symbol | Definition |
|---|---|
| $n_u, n_i$ | Number of users, items |
| $m$ | Number of interactions |
| $d$ | Embedding dimension (**value NOT STATED**) |
| $L$ | Number of layers (**value NOT STATED**) |
| $M$ | Shapley samples (**= 50**) |
| $f$ | Shapley refresh period in batches (**= 10**) |

Note that Eq. (24)'s memory term $O((n_u + n_i)d + m)$ omits context nodes, inconsistent with the three-type node set of §3.2.

**Reporting inconsistency:** §3.4 and §4.6 both state the overhead is *"~78% over baseline hypergraph GNN (2,001s vs 1,125s)"*, but 1,124.6 s is **HPCF's** time in Table 6 — HPCF being a specific published baseline, not "a baseline hypergraph GNN". The two framings are conflated. See catalogue item 38.

---

## 6. Complete Catalogue of Ambiguities and Gaps (63 items)

Every place where the paper is underspecified, self-contradictory, or arithmetically wrong. Organized by severity.

### 6.1 Blocking gaps — cannot write code without guessing (items 1–22)

| # | Gap |
|---|---|
| 1 | **Hyperedge construction is never specified.** No incidence matrix $H$, no rule mapping interactions to hyperedges, no definition of $A$ in terms of $H$, no statement of whether $D$ is vertex or hyperedge degree. The citation to HCCF [14] implies learnable hyperedges; the text implies a fixed interaction-derived sparse $A$. **Irreconcilable — a genuine architectural fork.** |
| 2 | **Embedding dimension $d$ — NOT STATED.** Symbol defined and used in Eq. (24); no value anywhere. |
| 3 | **Number of propagation layers $L$ — NOT STATED.** |
| 4 | **Learning rate $\eta$ — NOT STATED.** Symbol defined in Notation List only. |
| 5 | **$\lambda_{div}$, $\lambda_{ctx}$, $\lambda_{reg}$ — all NOT STATED.** Eq. (13), the entire training objective, is unusable as written. |
| 6 | **$\lambda_c$ (context weight, Eq. 12) — NOT STATED.** |
| 7 | **Dropout rate $p$ — NOT STATED.** "dropout $p$ on embeddings" is stated; the value is not. |
| 8 | **Layer-fusion coefficients $\alpha_l$ (Eq. 9) — NOT STATED**, and it is never said whether they are fixed uniform, tuned, or learned. |
| 9 | **$sim(i,j)$ is undefined.** Required by Eq. (16) (training loss), Eq. (23) (reported metric), and $Diversity(S)$ in Eq. (1). "Latent semantic space" is the only hint, and that reading would make the reported diversity metric model-dependent and non-comparable across baselines, invalidating the Diversity column of Table 3. |
| 10 | **$sim(u,i)$ is undefined.** Required by Eq. (2) and thus by the entire preference-aware mechanism — the paper's central named contribution. Only characterized as "historical or content-based affinities". |
| 11 | **$ContextScore(S)$ is never defined anywhere in the paper.** One of three terms of the characteristic function is simply absent. |
| 12 | **$Diversity(S)$ is never defined for a coalition.** Eq. (23) defines a metric; the paper never states Eq. (1) reuses it. |
| 13 | **$NDCG@20(S)$ for a coalition is never operationalized.** No procedure for producing a ranking from a coalition $S$, no statement of which users or ground truth are used, no statement of how a model "restricted to $S$" is constructed. |
| 14 | **The player→edge bridge is missing.** Eqs. (3)–(5) define $\hat\phi_j$ per player; Eq. (8) consumes $\hat\phi_{jk}$ per edge. No relation is ever given. |
| 15 | **$\mathcal{L}_{div}$ is non-differentiable as written** (depends on a top-$K$ sort). No relaxation, straight-through estimator, or surrogate is provided. |
| 16 | **$\sigma$ is never identified** in Eqs. (6), (7), or (10) — and the same symbol denotes the logistic function in Eq. (14). |
| 17 | **$l_i$ in Eq. (10) is never defined**, neither in §3.5 nor in the Notation List. |
| 18 | **$W_a$ output dimension unspecified.** Scalar gate vs. vector gate changes Eqs. (11)–(12) entirely. |
| 19 | **Number of training epochs — NOT STATED** (only patience = 20). |
| 20 | **Negative sampling fully underspecified:** ratio NOT STATED; popularity exponent NOT STATED; hard-negative criterion NOT STATED; hard-negative refresh period NOT STATED; with/without replacement NOT STATED. |
| 21 | **Shapley coalition sampling distribution unspecified:** subsets vs. permutation prefixes; shared vs. per-player samples; coalition size distribution. |
| 22 | **Variance-reduction parameters unspecified:** "light temporal smoothing" has no coefficient; "clipped the extremes" has no thresholds or percentiles. |

### 6.2 Equations mathematically broken as printed (items 23–30)

| # | Defect |
|---|---|
| 23 | **Eq. (11)** — $y_{ui} = (1+a_{ui})\cdot e_i^{\top}$: the RHS is a transposed vector, $e_u$ is entirely absent, and $y_{ui}$ is never used again anywhere in the paper. |
| 24 | **Eq. (12)** — adds the vector $g(c_{u,i})$ (a vector per Eq. 17) to the scalar $\langle e_u, e_i\rangle$. Undefined operation; the reduction is left to the implementer. |
| 25 | **Eq. (11) vs Eq. (12) conflict** — multiplicative gate vs. additive gate for $a_{ui}$, with no rule given for which applies at training or inference. |
| 26 | **Eq. (6) vs Eq. (7) conflict** — two different propagation rules are presented, with no statement that (7) supersedes (6). Eq. (7) has a trainable $W^{(l)}$, a self-term, and asymmetric Shapley normalization; Eq. (6) has none of these. |
| 27 | **Eq. (14)** — unnormalized sum over the full training set, no batch mean, making its scale relative to the mean-normalized Eqs. (15) and (17) dataset-size-dependent. |
| 28 | **Eq. (18)** — omits context embeddings $E_C$ from regularization without comment; and it is never clarified whether Adam `weight_decay` is additionally active (which would double-count regularization). |
| 29 | **Eq. (20)** — uses the exponential gain $2^{rel}-1$ while stating $rel$ is binary (equivalent to binary gain; harmless but redundant). |
| 30 | **Eqs. (15)/(16)** — the $K$ in the training-time $ILD$ is never tied to the evaluation $K$, and the ranked list $\mathcal{R}_u$ used during training is never defined. |

### 6.3 Protocol contradictions (items 31–44)

| # | Contradiction |
|---|---|
| 31 | **"70% train, 10% val, 20% test per user; leave-one-out"** — two mutually exclusive splitting protocols in a single clause (§3.8). |
| 32 | **§4.3 claims CV ensured "no user or item appeared in both the training and test sets"** — impossible for transductive CF with a per-user holdout, and directly contradicts the §3.8 protocol. |
| 33 | **Table 2 reports raw, unfiltered MovieLens-1M statistics** (1,000,209 interactions; 3,706 items) despite the stated `rating > 3` and 5-core filters. Applying `rating > 3` yields 575,281 positives. No post-preprocessing target exists to validate against. |
| 34 | **The Amazon-Book statistics exactly match the canonical LightGCN preprocessed split** (which has no ratings and no timestamps), making both `rating > 3` and "per-user temporal holdout" inapplicable to it. |
| 35 | **§3.1 calls $\mathcal{G}$ a bipartite graph over $\mathcal{U}\cup\mathcal{I}$; the Notation List calls it a user-item-context hypergraph.** |
| 36 | **Whether Eq. (4) or Eq. (5) supplies the weights in Eq. (8) is ambiguous.** Eq. (8) writes the unsuperscripted $\hat\phi$; the model's name and the $\lambda_{\text{pref}}$ default imply Eq. (5). |
| 37 | **Metrics are stated at $K\in\{5,10,20\}$ but only $K=20$ is ever reported.** |
| 38 | **"~78% overhead over baseline hypergraph GNN (2,001s vs 1,125s)"** — 1,124.6 s is HPCF's time in Table 6, a specific published baseline, not "a baseline hypergraph GNN". The two framings are conflated in both §3.4 and §4.6. |
| 39 | **Yelp2018 appears in Table 8 with no statistics, no preprocessing description, and no mention in §4.1.** Its dataset row is absent from Table 2. |
| 40 | **Ablation percentages don't match the values.** Amazon-Book w/o Attention: 0.0295 vs 0.0306 = 3.6%, listed 3.5%. w/o Diversity: 0.0288 vs 0.0306 = 5.9%, listed 5.8%. §4.7.3 also garbles the hypergraph row ("led to a 6.8% performance decline ... corresponding to drops of 6.8% and 8.9%"). |
| 41 | **Cold-start improvements are internally inconsistent.** Body text says +9.8% for both users and items (0.0606/0.0552 and 0.0571/0.0520); Table 8 says +10.9% and +9.6%. |
| 42 | **Cross-dataset improvements conflict with Table 3.** Table 8 gives +9.9% (ML) and +14.8% (Amazon); Table 3 yields +9.77% and +13.33%. |
| 43 | **Table 9's Holm thresholds are assigned in reverse order** of the sorted p-values, contradicting the paper's own stated $\alpha/(k-i+1)$ rule stated two paragraphs earlier. |
| 44 | **Table 4 fold values are exact arithmetic progressions** for all 7 models on both datasets, centered on the Table 3 means. Not usable as a variance reference and not a pattern real cross-validation produces. |

### 6.4 Notation inconsistencies between body and Notation List (items 45–51)

| # | Inconsistency |
|---|---|
| 45 | $e^{(l)}$ (body, Eq. 6) vs $X^{(\ell)}$ (Notation List) for stacked node embeddings. |
| 46 | $w_{jk}$ (body, Eq. 8) vs $\tilde\phi_{uv}$ (Notation List) for the normalized Shapley edge weight. |
| 47 | $a_{ui}$ (body, Eq. 10) vs $\alpha_{u,i}$ / $A_{att}$ (Notation List) for the attention weight. |
| 48 | $g(\cdot)$ and $\lambda_c$ (body, Eq. 12) vs $\psi(\cdot)$ and $w_c$ (Notation List) for the context function and its weight. |
| 49 | $\mathcal{L}_{rec}$ (body, Eq. 14) vs $\mathcal{L}_{rank}$ (Notation List) for the ranking term. |
| 50 | **$\alpha$ is used for three distinct things:** the NDCG weight in Eq. (1), the layer-fusion coefficients $\alpha_l$ in Eq. (9), and the significance level in Appendix A. Plus $\alpha_{u,i}$ for attention in the Notation List — a fourth use. |
| 51 | **$N$ / $\mathcal{N}$ is overloaded:** $N$ (Eqs. 1, 3) and $\mathcal{N}$ (Notation List) denote the cooperative-game player set, while $\mathcal{N}(j)$ / $\mathcal{N}(v)$ denotes a graph neighborhood in Eqs. (7)–(8). |

### 6.5 Things the paper is silent on entirely (items 52–63)

| # | Silence |
|---|---|
| 52 | Embedding initialization scheme and scale. |
| 53 | Whether $\hat\phi$ is detached from the computation graph (almost certainly yes given caching, but never stated). |
| 54 | Bias terms anywhere in the network. |
| 55 | Presence or absence of a nonlinearity in the final scorer. |
| 56 | Whether context embeddings are trained, frozen, or derived from features. |
| 57 | How multi-valued context (e.g. multiple movie genres) is pooled into $c_{u,i}$ and $\mathbf{e}_{c_{u,i}}$ — and what plays the role of context in Amazon-Book at all. |
| 58 | $|\mathcal{C}|$ (context vocabulary size) for any dataset. |
| 59 | The nature of the target $\mathbf{e}_{c_{u,i}}$ in Eq. (17): a separate frozen target table, a feature vector, or the same table $g$ maps into (in which case the loss is trivially zero and carries no signal). |
| 60 | Full-catalog vs. sampled-candidate evaluation — only obliquely implied by "candidates exclude training positives other than the evaluation target". |
| 61 | Baseline hyperparameters and implementation provenance (whether official author code was used for MF, NCF, LightGCN, RecDCL, HCCF, HPCF). The "shared grid" is referenced but never reported. |
| 62 | Learning-rate schedule, gradient clipping, mixed precision. |
| 63 | Any code, data, or artifact availability whatsoever. |

---

## 7. Full Equation Index (all 24)

| Eq. | Section | Purpose | Fully specified? |
|---|---|---|---|
| (1) | 3.3 | Coalition value $v(S) = \alpha\,NDCG@20(S) + \beta\,Diversity(S) + \gamma\,ContextScore(S)$ | ❌ All three terms undefined; weights stated |
| (2) | 3.3 | Preference-weighted value $v_{pref}(S) = v(S) + \lambda_{pref}\sum sim(u,i)$ | ❌ $sim(u,i)$ undefined; $\lambda_{pref}$ stated |
| (3) | 3.4 | Exact Shapley value | ✅ Standard closed form |
| (4) | 3.4 | Monte Carlo Shapley estimator | ⚠️ Sampling distribution unspecified |
| (5) | 3.4 | Preference-aware Monte Carlo estimator | ⚠️ Same, plus depends on Eq. (2) |
| (6) | 3.5 | Symmetric-normalized propagation | ❌ $A$, $D$, $\sigma$ all unspecified; conflicts with (7) |
| (7) | 3.5 | Shapley-weighted message passing | ❌ $\sigma$, $W^{(l)}$ shape, $\mathcal{N}(j)$ unspecified |
| (8) | 3.5 | Shapley edge-weight normalization | ❌ $\hat\phi_{jk}$ indexing never bridged to $\hat\phi_j$ |
| (9) | 3.5 | Layer-wise fusion | ❌ $\alpha_l$ and $L$ unspecified |
| (10) | 3.5 | Interaction-level attention gate | ❌ $l_i$ undefined, $W_a$ shape and $\sigma$ unspecified |
| (11) | 3.5 | Intermediate score | ❌ **Dimensionally broken; never used again** |
| (12) | 3.5 | Final context-aware prediction | ❌ **Dimensionally broken; $\lambda_c$ unspecified; conflicts with (11)** |
| (13) | 3.6 | Composite loss | ❌ All three $\lambda$ unspecified |
| (14) | 3.6 | BPR ranking loss | ⚠️ Unnormalized; $\mathcal{D}$ construction unspecified |
| (15) | 3.6 | Diversity regularizer | ❌ **Non-differentiable as written**; $\mathcal{R}_u$ undefined |
| (16) | 3.6 | ILD (training) | ❌ $sim(i_k,i_l)$ undefined; $K$ untied |
| (17) | 3.6 | Context alignment loss | ❌ Target $\mathbf{e}_{c_{u,i}}$ nature unspecified |
| (18) | 3.6 | Embedding regularization | ⚠️ Excludes $E_C$; interaction with Adam weight_decay unclear |
| (19) | 3.7 | NDCG@20 | ✅ Standard |
| (20) | 3.7 | Per-user DCG@20 | ✅ Standard (IDCG form implied) |
| (21) | 3.7 | Recall@20 | ⚠️ Macro vs. micro averaging unclear |
| (22) | 3.7 | Catalog coverage | ⚠️ List length $K$ implied not stated |
| (23) | 3.7 | Intra-list diversity (reported metric) | ❌ $sim(i,j)$ undefined |
| (24) | 4.6 | Training complexity per epoch | ✅ Complete as an asymptotic statement |

Tally: **3 fully specified, 6 partially specified, 15 blocked by undefined quantities or dimensional defects.**

---

## 8. Recommended Defaults — IMPLEMENTER ASSUMPTIONS, NOT PAPER CONTENT

> ### ⚠️ PROVENANCE WARNING
>
> **Nothing in this section comes from the paper.** Every value below is an implementer decision, chosen from the conventional settings of this model family (the LightGCN → HCCF → HPCF lineage that DyHuCoG positions itself against). They exist so that a reimplementation can proceed at all.
>
> **Record each of these in code as an explicitly flagged assumption** — e.g. a config file with a `provenance: assumption` field per key, or a module-level constant block with a comment citing this section — so that the origin of every choice remains auditable and no assumption can later be mistaken for a paper claim.

### 8.1 Architecture

| Item | Recommended default | Rationale | Provenance |
|---|---|---|---|
| Embedding dimension $d$ | **64** | Universal default across LightGCN, HCCF, HPCF; keeps the ~4.4 GB memory figure of Table 6 plausible. | **ASSUMPTION** |
| Number of layers $L$ | **3** | LightGCN/HCCF standard; $L=3$ is also where Eq. (24)'s $(L+1)$ factor matches the reported runtime ratios best. | **ASSUMPTION** |
| Hypergraph construction | **One hyperedge per observed interaction, incident on $\{u, i, c_{u,i}\}$; star (bipartite) expansion to build $A$; $D = D_v$ vertex degree.** Build $H \in \{0,1\}^{(|\mathcal{U}|+|\mathcal{I}|+|\mathcal{C}|)\times|\mathcal{E}|}$ from interactions, then $A = \begin{bmatrix}0 & H \\ H^\top & 0\end{bmatrix}$. | Matches the Notation List's "$\mathcal{E}$ = interactions (hyperedges)" and makes the "w/o Hypergraph → collapse to bipartite" ablation well-defined (drop the context row block from $H$). Rejects the HCCF learned-hyperedge reading because the paper describes a fixed sparse $A$. | **ASSUMPTION — resolves catalogue item 1** |
| Self-loops | **None** (Eq. 7's $W^{(l)}e_j^{(l)}$ self-term already provides one). | Avoids double-counting. | **ASSUMPTION** |
| Layer-fusion coefficients $\alpha_l$ | **Uniform $1/(L+1)$** | LightGCN convention; the paper's "lightweight" framing supports the non-learned choice. | **ASSUMPTION** |
| Activation $\sigma$ in Eqs. (6)–(7) | **LeakyReLU (negative slope 0.2)**, or **identity** if reproducing LightGCN-style lightweight propagation | HCCF uses LeakyReLU. Consider identity as an ablation, since LightGCN showed nonlinearity hurts in CF propagation. | **ASSUMPTION** |
| Activation $\sigma$ in Eq. (10) | **Logistic sigmoid** | Required for the $(1+a_{ui})$ form of Eq. (11) to be a sensible gate in $(1,2)$. | **ASSUMPTION** |
| $W^{(l)}$ shape | **$\mathbb{R}^{d\times d}$**, no bias | Only dimensionally consistent reading of Eq. (7). | **ASSUMPTION** |
| $W_a$ shape | **$\mathbb{R}^{1\times 3d}$ (scalar gate), no bias** | Supported by the Notation List's "$\alpha_{u,i}$ — Attention weight over user-item relation" (a scalar) and by Eq. (12) adding $a_{ui}$ to a scalar. | **ASSUMPTION — resolves catalogue item 18** |
| $l_i$ in Eq. (10) | **The item's context/genre embedding $\mathbf{e}_{c_{\cdot,i}}$**, mean-pooled over the item's genres | Only reading consistent with "conditioning the user, item, and genre/context embeddings". | **ASSUMPTION — resolves catalogue item 17** |
| Embedding initialization | **Xavier/Glorot normal**, or $\mathcal{N}(0, 0.01^2)$ | LightGCN/HCCF standard. | **ASSUMPTION** |
| Bias terms | **None anywhere** | Matches the printed equations. | **ASSUMPTION** |
| Propagation rule to use | **Eq. (7) only. Discard Eq. (6).** | Eq. (6) has no Shapley weighting and no parameters, so it cannot be the model that the "w/o Shapley → uniform weights" ablation degrades to. Eq. (6) is best read as expository background. | **ASSUMPTION — resolves catalogue item 26** |

### 8.2 Scoring

| Item | Recommended default | Rationale | Provenance |
|---|---|---|---|
| Scoring equation | **Eq. (12) only. Discard Eq. (11).** | Eq. (12) is labelled "The final context-aware prediction" and is the one referenced by the BPR loss (Eq. 14). $y_{ui}$ from Eq. (11) is never used again. | **ASSUMPTION — resolves catalogue item 25** |
| Reduction of $g(c_{u,i})$ to a scalar | **$\langle e_u,\, g(c_{u,i})\rangle$** | Makes the context term user-personalized, which is the point of a context-aware score, and keeps $g(\cdot)$ a vector so Eq. (17) remains well-formed. | **ASSUMPTION — resolves catalogue item 24** |
| $\lambda_c$ | **0.1** | Small enough that the context term refines rather than dominates the inner product; consistent with context being an auxiliary signal worth 8.2% of NDCG (Table 5). | **ASSUMPTION** |

### 8.3 Similarity functions

| Item | Recommended default | Rationale | Provenance |
|---|---|---|---|
| $sim(i,j)$ (Eqs. 16, 23) | **Cosine similarity on a FIXED content/genre vector, not on learned embeddings.** For MovieLens-1M use the 18-dim multi-hot genre vector. | Keeps the reported diversity metric model-independent and therefore comparable across baselines. Using learned embeddings would make the Diversity column of Table 3 incomparable across models and invalidate the comparison. **This is the single most important assumption in this document.** | **ASSUMPTION — resolves catalogue item 9** |
| $sim(u,i)$ (Eq. 2) | **Cosine between the user's mean historical item content vector and item $i$'s content vector**, i.e. $\cos\!\bigl(\frac{1}{|\mathcal{I}_u|}\sum_{j\in\mathcal{I}_u} \mathbf{x}_j,\; \mathbf{x}_i\bigr)$ where $\mathbf{x}$ is the genre vector. | Directly instantiates the paper's phrase "historical or content-based affinities", and is computable once offline. | **ASSUMPTION — resolves catalogue item 10** |
| Amazon-Book content vectors | **Derive a pseudo-context by clustering item co-occurrence into $|\mathcal{C}| = 20$ latent groups**, or omit context and report it as a deviation. | The paper never says what context is for Amazon-Book, and the canonical preprocessed split has no metadata. Any choice here is a deviation and must be documented. | **ASSUMPTION — resolves catalogue item 57** |

### 8.4 Cooperative game

| Item | Recommended default | Rationale | Provenance |
|---|---|---|---|
| $ContextScore(S)$ | **Mean over interactions in $S$ of the cosine alignment $\cos(g(c_{u,i}), \mathbf{e}_{c_{u,i}})$**, i.e. $1 - $ normalized Eq. (17) residual, mapped to $[0,1]$. | The only quantity in the paper that measures "contextual alignment"; reusing it keeps $v(S)$'s three terms on a common $[0,1]$ scale. | **ASSUMPTION — resolves catalogue item 11** |
| $Diversity(S)$ | **Eq. (23) applied to the top-$K$ lists produced by the $S$-restricted model** | Simplest consistent reading; keeps the term in $[0,1]$. | **ASSUMPTION — resolves catalogue item 12** |
| $NDCG@20(S)$ | **Eq. (19) evaluated on a fixed sampled probe set of users (e.g. 200 validation users, matching the paper's stated MSE-reference subset size), scoring with embeddings masked to zero for players outside $S$.** | Masking is the cheapest well-defined notion of "produced using only the entities in $S$". Reusing the 200-user subset size the paper cites for its MSE reference keeps the cost bounded. | **ASSUMPTION — resolves catalogue item 13** |
| Estimator used for Eq. (8) | **Eq. (5), the preference-aware estimator $\hat\phi_j^{pref}$** | The model is named preference-aware and $\lambda_{pref}$ has a stated default. | **ASSUMPTION — resolves catalogue item 36** |
| Player→edge mapping | **$\hat\phi_{jk} \leftarrow \hat\phi_k^{pref}$** (the message weight is the Shapley value of the sending node), then row-normalize per Eq. (8). | Simplest reading that makes Eq. (8) well-defined with per-player estimates. Document the alternative (true interaction-level estimation) as a variant to ablate. | **ASSUMPTION — resolves catalogue item 14** |
| Coalition sampling | **Permutation sampling: draw $M=50$ uniform random permutations of $\mathcal{N}$, shared across all players; $S_m$ is the prefix preceding $j$ in permutation $m$.** | This is the standard unbiased scheme and is what "samples coalitions (equivalently, permutations)" points at; sharing permutations makes total cost $O(M|\mathcal{N}|)$ rather than $O(M|\mathcal{N}|^2)$. | **ASSUMPTION — resolves catalogue item 21** |
| Non-negativity of $\hat\phi$ | **Shift-and-clip: $\hat\phi \leftarrow \mathrm{clip}(\hat\phi,\, q_{01},\, q_{99})$ then $\hat\phi \leftarrow \hat\phi - \min(\hat\phi) + \varepsilon$ with $\varepsilon = 10^{-8}$** before Eq. (8). | Eq. (8) is ill-behaved with negative or near-zero-sum weights, which real Shapley estimates will produce. Instantiates the paper's "clipped the extremes". | **ASSUMPTION — resolves catalogue item 22** |
| Temporal smoothing | **EMA with $\rho = 0.9$: $\hat\phi_t \leftarrow 0.9\,\hat\phi_{t-1} + 0.1\,\hat\phi_{\text{new}}$** | Instantiates "light temporal smoothing"; $\rho=0.9$ over a 10-batch refresh gives a ~100-batch effective memory. | **ASSUMPTION** |
| Gradient treatment | **Estimate under `torch.no_grad()`; store as a detached, registered buffer.** | Values are refreshed every 10 batches and cached, so they cannot carry gradients coherently. | **ASSUMPTION — resolves catalogue item 53** |

### 8.5 Loss

| Item | Recommended default | Rationale | Provenance |
|---|---|---|---|
| $\lambda_{div}$ | **0.1** | Diversity is worth 5.8% of NDCG (Table 5) — a real but secondary contribution, consistent with a small weight. | **ASSUMPTION** |
| $\lambda_{ctx}$ | **0.1** | Same reasoning; context is worth 8.2%, so consider 0.1–0.2 in a sweep. | **ASSUMPTION** |
| $\lambda_{reg}$ | **1e-4** | Universal LightGCN/HCCF default for BPR embedding regularization. | **ASSUMPTION** |
| $\mathcal{L}_{rec}$ normalization | **Batch mean** (divide Eq. 14 by $|\mathcal{B}|$) | Makes the $\lambda$ values dataset-size-independent, which the unnormalized Eq. (14) does not. **Deviation from the printed equation — document it.** | **ASSUMPTION — resolves catalogue item 27** |
| $\mathcal{L}_{div}$ differentiable surrogate | **Compute the top-$K$ candidate set with `torch.no_grad()` (detached indices), then evaluate the pairwise $1-sim$ term differentiably over the *embeddings* of those items.** Alternatively apply ILD to in-batch sampled positives to avoid a full-catalog sort every step. | Gives a usable gradient path while keeping the list construction non-differentiable, as it must be. **Deviation — document it.** | **ASSUMPTION — resolves catalogue item 15** |
| $K$ in the training ILD | **20**, matching the evaluation cutoff | Simplest consistent choice. | **ASSUMPTION — resolves catalogue item 30** |
| $\mathbf{e}_{c_{u,i}}$ target in Eq. (17) | **A separate, frozen target built from the raw content features** (e.g. the L2-normalized multi-hot genre vector, linearly projected to $d$ by a fixed random matrix), distinct from the trainable table $g$ maps into. | If the target were the same trainable table, Eq. (17) collapses to zero and carries no signal. | **ASSUMPTION — resolves catalogue item 59** |
| Regularization mechanism | **Use $\mathcal{L}_{reg}$ (Eq. 18) with Adam `weight_decay=0`**, and extend Eq. (18) to include $E_C$. | Avoids double-counting; including context embeddings is more principled than the printed omission. **Deviation — document it.** | **ASSUMPTION — resolves catalogue item 28** |

### 8.6 Optimization

| Item | Recommended default | Rationale | Provenance |
|---|---|---|---|
| Learning rate $\eta$ | **1e-3** | Adam default and the LightGCN/HCCF/HPCF standard. | **ASSUMPTION** |
| Adam $\beta_1,\beta_2,\epsilon$ | **0.9, 0.999, 1e-8** (PyTorch defaults) | No reason to deviate. | **ASSUMPTION** |
| Adam `weight_decay` | **0** | Regularization handled by $\mathcal{L}_{reg}$. | **ASSUMPTION** |
| Dropout $p$ | **0.1 on embeddings** | Light regularization consistent with "dropout $p$ on embeddings". | **ASSUMPTION** |
| Max epochs | **200** | Sufficient headroom for patience-20 early stopping on validation NDCG@20 to trigger first. | **ASSUMPTION — resolves catalogue item 19** |
| LR schedule | **None** (constant) | Not mentioned; constant is the family default. | **ASSUMPTION** |
| Gradient clipping | **None** | Not mentioned. | **ASSUMPTION** |
| Mixed precision | **Off** (fp32) | Keeps the Table 6 memory figures comparable. | **ASSUMPTION** |

### 8.7 Negative sampling

| Item | Recommended default | Rationale | Provenance |
|---|---|---|---|
| Ratio | **1 negative per positive** | BPR standard; Eq. (14) is written over single triples $(u,i^+,i^-)$. | **ASSUMPTION — resolves catalogue item 20** |
| Distribution | **$p(i) \propto \mathrm{pop}(i)^{0.75}$** | The canonical "popularity-aware" sampler (word2vec/BPR convention). | **ASSUMPTION** |
| Hard negatives | **Every 10 batches (aligned with the Shapley refresh), replace 25% of the negative pool with the highest-scoring non-interacted items from a 100-item random candidate pool per user.** | Instantiates "periodic hard-negative refreshes" cheaply, and aligning the period with $f=10$ keeps one refresh cadence in the training loop. | **ASSUMPTION** |
| Exclusions | **Exclude all known positives for the user, including validation and test positives, from the negative pool.** | Prevents label leakage into the training signal. | **ASSUMPTION** |
| Replacement | **With replacement** | Standard, cheap. | **ASSUMPTION** |

### 8.8 Data and evaluation

| Item | Recommended default | Rationale | Provenance |
|---|---|---|---|
| Split protocol | **Per-user temporal holdout 70/10/20, sorting each user's interactions by timestamp. Discard the "leave-one-out" clause.** | 70/10/20 is the more specific and more usable of the two contradictory statements, and it is the one that makes Recall@20 meaningful (leave-one-out with a single test item caps Recall behavior). Document the deviation. | **ASSUMPTION — resolves catalogue item 31** |
| Filter order | **Apply `rating > 3` binarization FIRST, then iterative 5-core on users and items.** | Filtering after binarization is the standard order and yields a graph where every retained node genuinely has 5 positive interactions. Report the resulting statistics alongside Table 2 and expect them to differ. | **ASSUMPTION — resolves catalogue item 33** |
| Expected ML-1M post-filter scale | **~575,281 positives before 5-core; expect ~570k after** | For sanity-checking the pipeline, since Table 2's 1,000,209 is pre-filter. | **ASSUMPTION (derived)** |
| Amazon-Book | **Use the canonical LightGCN preprocessed split as-is (52,643 / 91,599 / 2,984,108) and accept its native 80/20 random split**, documenting that the paper's temporal 70/10/20 and `rating > 3` cannot be applied. | Matching the paper's reported statistics and matching its stated protocol are mutually exclusive here; matching the statistics is the more verifiable choice. | **ASSUMPTION — resolves catalogue item 34** |
| Evaluation candidates | **Full-catalog ranking with all training (and validation, when testing) positives masked to $-\infty$.** | Matches "candidates exclude training positives other than the evaluation target" and is the standard for the LightGCN/HCCF/HPCF numbers being compared against. | **ASSUMPTION — resolves catalogue item 60** |
| Recall averaging | **Macro (per-user then mean)** | Matches §3.7's "report user-averaged scores". | **ASSUMPTION** |
| IDCG form | **$\sum_{k=1}^{\min(20,\,|\text{test}_u|)} 1/\log_2(k+1)$** | Standard binary-relevance ideal DCG. | **ASSUMPTION** |
| Coverage list length | **20** | Matches the reported cutoff. | **ASSUMPTION** |
| Seeds | **{42, 43, 44, 45, 46}** | **STATED by the paper** — not an assumption. | Paper, §3.8 |
| Report at | **$K \in \{5, 10, 20\}$** | **STATED by the paper**; report all three even though the paper only shows 20. | Paper, §3.8 |

### 8.9 Consolidated assumption config block

A single reference listing for a config file. **Every key here is an ASSUMPTION unless marked `[PAPER]`.**

```yaml
# provenance: ASSUMPTION unless marked [PAPER]
model:
  embedding_dim: 64              # ASSUMPTION
  num_layers: 3                  # ASSUMPTION
  layer_fusion: uniform          # ASSUMPTION (alpha_l = 1/(L+1))
  activation_prop: leaky_relu    # ASSUMPTION (slope 0.2)
  activation_gate: sigmoid       # ASSUMPTION
  attention_out_dim: 1           # ASSUMPTION (scalar gate, W_a in R^{1x3d})
  bias: false                    # ASSUMPTION
  self_loops: false              # ASSUMPTION
  scoring_equation: eq12         # ASSUMPTION (discard Eq. 11)
  propagation_equation: eq7      # ASSUMPTION (discard Eq. 6)
  context_reduction: dot_with_user  # ASSUMPTION (resolves broken Eq. 12)
  lambda_c: 0.1                  # ASSUMPTION

hypergraph:
  hyperedge_rule: one_per_interaction_uic   # ASSUMPTION
  expansion: star                           # ASSUMPTION
  degree_matrix: vertex                     # ASSUMPTION
  learnable_hyperedges: false               # ASSUMPTION (rejects HCCF reading)

game:
  alpha: 0.60                    # [PAPER] §3.3
  beta: 0.25                     # [PAPER] §3.3
  gamma: 0.15                    # [PAPER] §3.3
  lambda_pref: 0.20              # [PAPER] §3.3
  M: 50                          # [PAPER] §3.4, Table 7
  refresh_every_batches: 10      # [PAPER] §3.4, §4.6
  estimator: preference_aware    # ASSUMPTION (Eq. 5 over Eq. 4)
  sampling: shared_permutations  # ASSUMPTION
  player_to_edge: sender_value   # ASSUMPTION (phi_jk <- phi_k)
  clip_quantiles: [0.01, 0.99]   # ASSUMPTION
  ema_rho: 0.9                   # ASSUMPTION
  probe_users: 200               # ASSUMPTION (mirrors paper's MSE subset size)
  detach: true                   # ASSUMPTION

loss:
  lambda_div: 0.1                # ASSUMPTION
  lambda_ctx: 0.1                # ASSUMPTION
  lambda_reg: 1.0e-4             # ASSUMPTION
  bpr_reduction: mean            # ASSUMPTION (deviates from unnormalized Eq. 14)
  ild_K: 20                      # ASSUMPTION
  ild_topk_detached: true        # ASSUMPTION (differentiable surrogate)
  reg_includes_context: true     # ASSUMPTION (extends Eq. 18)

similarity:
  item_item: cosine_on_genre_multihot   # ASSUMPTION - most important choice
  user_item: cosine_user_profile_genre  # ASSUMPTION

optim:
  optimizer: adam                # [PAPER] §3.6, §3.8
  lr: 1.0e-3                     # ASSUMPTION
  weight_decay: 0.0              # ASSUMPTION
  dropout: 0.1                   # ASSUMPTION
  batch_size: 2048               # [PAPER] §3.4, §4.6
  eval_batch_size: 1024          # [PAPER] §3.8
  max_epochs: 200                # ASSUMPTION
  early_stop_metric: ndcg@20     # [PAPER] §3.6, §3.8
  early_stop_patience: 20        # [PAPER] §3.8
  grad_clip: null                # ASSUMPTION
  amp: false                     # ASSUMPTION

negatives:
  ratio: 1                       # ASSUMPTION
  popularity_exponent: 0.75      # ASSUMPTION
  hard_negative_every: 10        # ASSUMPTION
  hard_negative_fraction: 0.25   # ASSUMPTION
  exclude_val_test_positives: true  # ASSUMPTION

data:
  binarize_threshold: 3          # [PAPER] §3.8 (rating > 3)
  k_core: 5                      # [PAPER] §3.8 (minimum 5 interactions)
  filter_order: binarize_then_kcore  # ASSUMPTION
  split: per_user_temporal_70_10_20  # [PAPER] §3.8 (leave-one-out clause discarded — ASSUMPTION)
  eval_protocol: full_catalog_masked # ASSUMPTION
  seeds: [42, 43, 44, 45, 46]     # [PAPER] §3.8, §4.1
  cutoffs: [5, 10, 20]            # [PAPER] §3.8
  workers: 4                      # [PAPER] §3.8
```

### 8.10 Expectation setting

Given the density of gaps, **treat the paper's reported numbers as aspirational rather than reproducible targets.** In particular:

- NDCG@20 = 0.2775 on MovieLens-1M and 0.0306 on Amazon-Book depend on at least ten unstated values, so an exact match would be coincidental.
- The Table 3 **ordering** of baselines (MF < NCF < LightGCN < RecDCL < HCCF < HPCF < DyHuCoG) is a more meaningful reproduction target than the absolute values.
- The **ablation ordering** (Context > Hypergraph > Shapley > Diversity > Attention) is a reasonable qualitative target, and the most useful sanity check that the components are wired correctly.
- Table 4 must not be used as a variance reference (catalogue item 44).
- The reproducible contribution is the **architecture**: Shapley-weighted hypergraph message passing (Eqs. 5, 7, 8) plus an independent interaction-level attention gate (Eq. 10) sitting outside the propagation stack, yielding two separable per-player importance signals.

---

## 9. Verbatim Notation List from the Paper

Reproduced exactly as printed, for reference. Note the divergences from the body text flagged in catalogue items 45–51.

| Symbol | Description |
|---|---|
| $\mathcal{U}$ | Set of users |
| $\mathcal{I}$ | Set of items |
| $\mathcal{C}$ | Set of contexts (e.g., time, session, genre) |
| $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ | User-item-context hypergraph |
| $\mathcal{E}$ | Set of observed interactions (hyperedges) in $\mathcal{G}$ |
| $u, i$ | User and item indices |
| $c_{u,i}$ | Context vector associated with a $(u,i)$ interaction |
| $f(u,i,c_{u,i})$ | Context-aware scoring function for ranking items for user $u$ |
| $K$ | Cutoff for top-$K$ evaluation (e.g., $K = 20$) |
| $\mathcal{N}$ | Player set (users, items, contexts) in the cooperative game |
| $S \subseteq \mathcal{N}$ | Coalition (subset of players) |
| $v(S)$ | Coalition value (utility combining accuracy, diversity, and context) |
| $v_{\text{pref}}(S)$ | Preference-weighted coalition value |
| $\phi_j$ | Shapley value for player $j$ (exact) |
| $\hat\phi_j$ | Monte Carlo estimate of Shapley value for player $j$ |
| $\hat\phi_j^{pref}$ | Preference-aware Shapley value for player $j$ |
| $M$ | Number of Monte Carlo samples / permutations |
| $\mathcal{S}'$ | Random subset of $\mathcal{N}$ used for sampling |
| $A$, $D$ | Hypergraph adjacency and (hyper)degree matrices |
| $X^{(\ell)}$ | Stacked node embeddings at GNN layer $\ell$ |
| $W^{(\ell)}$ | Trainable weight matrix at layer $\ell$ |
| $L$ | Number of GNN layers |
| $\mathcal{N}(v)$ | Neighborhood of node $v$ in the hypergraph |
| $\tilde\phi_{uv}$ | Normalized Shapley-derived (hyper)edge weight used in message passing |
| $\alpha_{u,i}$ | Attention weight over user-item (and optional context) relation |
| $A_{att}$ | Attention matrix/tensor over relations |
| $g$ | Context/genre embedding vector |
| $\psi(\cdot)$ | Context-embedding function |
| $w_c$ | Context-alignment weight/temperature |
| $\mathcal{L}$ | Composite objective (ranking + diversity + context + regularization) |
| $\mathcal{L}_{rank}$ | Ranking term (e.g., BPR; pairwise positive over negative) |
| $\mathcal{L}_{div}$ | Diversity regularizer (e.g., intra-list diversity, ILD) |
| $\mathcal{L}_{ctx}$ | Context-alignment loss |
| $\lambda$ | Weights for multi-objective loss components |
| $\eta$ | Learning rate |
| $B$ | Batch size |
| $p$ | Dropout rate |
| Coverage | Catalog coverage across recommended lists |
| ANN | Approximate Nearest Neighbor |

---

*End of specification. Sections 1–7 and 9 are extraction from the paper; Section 8 is implementer assumptions. The warning block at the top of this file is the summary judgement.*
