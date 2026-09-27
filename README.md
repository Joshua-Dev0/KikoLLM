# KikoLLM

A compact, highly optimized, **decoder-only Transformer language model** designed for local training on consumer hardware. KikoLLM leverages modern architectural components found in cutting-edge LLMs (like LLaMA and Mistral) scaled down to an ultra-efficient size.

---

## 📊 Model Specifications

| Spec | Recommendation |
| :--- | :--- |
| **Architecture** | Decoder-only Transformer |
| **Parameters** | **~10–20M** |
| **Context length** | **256 tokens** |
| **Vocabulary** | **~8,000 tokens** |
| **`d_model`** | **256** |
| **Transformer blocks** | **6** |
| **Attention heads** | **8** |
| **Head dimension** | 32 |
| **FFN dimension** | **~512** |
| **FFN activation** | **SwiGLU** |
| **Positional encoding** | **RoPE** (Rotary Position Embedding) |
| **Normalization** | **RMSNorm** |
| **Attention** | **Causal self-attention** |
| **Output** | Vocabulary logits |
| **Weight tying** | **Yes** — Shared embedding and output linear weights |
| **Precision** | FP32 initially; FP16/BF16 later for acceleration |
| **Optimizer** | AdamW |
| **Training Objective** | Next-token prediction |
| **Hardware Target** | NVIDIA RTX 4060 (8 GB) |
| **Training Time** | **<12 hours** |

---

## 🏗️ Architecture & Pipeline Flow

The model processes inputs sequentially through the following pipeline during forward propagation:

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Embedding
 ↓
6 × Transformer Block
 │
 ├─ Pre-RMSNorm
 ├─ Multi-Head Causal Self-Attention
 │   ├─ Q, K, V Projections
 │   ├─ RoPE (Rotary Position Embeddings)
 │   ├─ Causal Masking
 │   └─ Attention Calculation: softmax(QKᵀ / √d_head)
 │
 ├─ Residual Connection
 │
 ├─ Pre-RMSNorm
 ├─ SwiGLU FFN (Feed-Forward Network)
 │
 └─ Residual Connection
 ↓
Final RMSNorm
 ↓
Linear Layer → Vocabulary Logits (Weight-tied)
 ↓
Softmax (Loss calculation / Generation sampling)
 ↓
Next-Token Prediction
```

---

## 📐 Exact Tensor Dimensions

The following dimensions assume a fully padded or maximum sequence context of **256 tokens** using the **(Batch, Sequence Length, Hidden Dimension)** convention. Tensor shapes below are isolated to a single sequence `(seq_len, dim)` for structural clarity.

### 1. Inputs & Embeddings
*   **Input Sequence (`X`):** `(256, 256)` where `seq_len = 256` and `d_model = 256`.
*   **Embedding Matrix:** `(8000, 256)` based on `vocab_size = 8000` and `d_model = 256`.

### 2. Causal Self-Attention Layer
*   **Total Project Tensors:** 
    *   Query (`Q`): `(256, 256)`
    *   Key (`K`): `(256, 256)`
    *   Value (`V`): `(256, 256)`
*   **Per-Head Projections (`num_heads = 8`, `head_dim = 32`):**
    *   Head `Q`: `(256, 32)`
    *   Head `K`: `(256, 32)`
    *   Head `V`: `(256, 32)`
*   **Attention Matrix (`Q @ K.T`):** `(256, 256)` score matrix per head before causal masking and Softmax.

### 3. SwiGLU Feed-Forward Network (FFN)
*   **Layer Weights:**
    *   `W_gate` (Gating projection): `(512, 256)`
    *   `W_up` (Up projection): `(512, 256)`
    *   `W_down` (Down projection): `(256, 512)`
*   **Layer Operations:**
    \[\text{gate} = \text{SiLU}(X W_{\text{gate}}^T) \quad \rightarrow \quad (256, 512)\]
    \[\text{value} = X W_{\text{up}}^T \quad \rightarrow \quad (256, 512)\]
    \[\text{hidden} = \text{gate} \odot \text{value} \quad \rightarrow \quad (256, 512)\]
    \[\text{output} = \text{hidden} W_{\text{down}}^T \quad \rightarrow \quad (256, 256)\]
