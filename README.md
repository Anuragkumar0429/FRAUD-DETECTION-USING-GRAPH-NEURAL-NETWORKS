<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=Intelligent%20Fraud%20Detection&fontSize=38&fontColor=ffffff&fontAlignY=38&desc=Graph%20Neural%20Networks%20%7C%20PyTorch%20Geometric%20%7C%20Heterogeneous%20GAT&descAlignY=58&descSize=17&animation=fadeIn" />

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&pause=1000&color=A78BFA&center=true&vCenter=true&width=700&lines=Heterogeneous+Graph+Attention+Network+🧠;Fraud+Ring+Detection+via+Graph+Structure+🕵️;590K+Transactions+%7C+AUC-ROC+~0.94;GNNExplainer+%2B+Streamlit+Dashboard;PyTorch+Geometric+%7C+IEEE-CIS+Dataset)](https://git.io/typing-svg)

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

</div>

---

## 🎯 Project Overview

This project converts the tabular IEEE-CIS fraud dataset into a **heterogeneous graph** and trains a **Graph Attention Network (GAT)** to detect fraudulent transactions. The key insight: fraudsters **reuse** cards, devices, and email domains — patterns invisible to row-by-row ML but obvious in graph structure.

### Why Graphs Beat Tables for Fraud Detection

| Tabular ML | Graph Neural Network |
|------------|---------------------|
| Sees each transaction in isolation | Sees transaction in the context of its neighbors |
| Can't detect fraud rings | Explicitly models shared card/device/email linkages |
| Feature engineering required | Learns structural patterns automatically |
| No transductive reasoning | Multi-hop reasoning across 2-3 hops |

---

## 🏗️ Graph Architecture

```
Nodes:
  ┌─────────────────┐   ┌──────────┐   ┌───────────┐
  │   Transaction   │   │   Card   │   │  Email    │
  │  (590k nodes)   │   │  entity  │   │  domain   │
  └────────┬────────┘   └──────────┘   └───────────┘
           │ uses_card  ────────────────────────────▶ Card
           │ uses_email ────────────────────────────▶ Email
           │ uses_device ───────────────────────────▶ Device
           │ uses_addr  ────────────────────────────▶ Address
           │ same_card  ────────────────────────────▶ Transaction

How Fraud Rings Are Caught (3-hop example):
  TX_A → CARD_123 ← TX_B → DEVICE_456 ← TX_C
  A 3-layer GAT aggregates signals from the entire ring.
  A single confirmed fraud node "contaminates" its neighbors.
```

---

## 📁 Project Structure

```
fraud_gnn/
│
├── data/
│   ├── train_transaction.csv       ← Kaggle download
│   └── train_identity.csv          ← Kaggle download
│
├── models/
│   ├── gnn_model.py               ← FraudGNN (GAT) + FraudGraphSAGE
│   ├── explainability.py          ← GNNExplainer + gradient importance
│   └── saved/                     ← Training artifacts (auto-created)
│       ├── best_model.pt
│       ├── encoders.pkl
│       ├── scaler.pkl
│       └── model_meta.pkl
│
├── utils/
│   ├── preprocessing.py           ← Load, clean, encode, normalize
│   ├── graph_builder.py           ← HeteroData graph construction
│   └── visualization.py           ← NetworkX, PyVis, Plotly charts
│
├── app.py                         ← Streamlit dashboard
├── train.py                       ← Full training pipeline
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Setup

```bash
git clone https://github.com/AnuragKumar0429/fraud-gnn.git
cd fraud-gnn

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Check your torch version
python -c "import torch; print(torch.__version__)"

# Install matching PyG wheels (example: torch 2.11.0 + CPU)
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.11.0+cpu.html

# For CUDA 12.1
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.11.0+cu121.html

pip install torch-geometric
```

### 2. Download Dataset

```bash
# Option A: Kaggle CLI
pip install kaggle
kaggle competitions download -c ieee-fraud-detection -p data/
cd data && unzip ieee-fraud-detection.zip

# Option B: Manual → https://www.kaggle.com/c/ieee-fraud-detection/data
# Place train_transaction.csv and train_identity.csv in data/
```

### 3. Train

```bash
# Full training (GPU recommended)
python train.py \
    --transaction data/train_transaction.csv \
    --identity    data/train_identity.csv    \
    --epochs      50                         \
    --model       gat                        \
    --hidden      128

# Quick prototype (subsample 50k transactions)
python train.py \
    --transaction data/train_transaction.csv \
    --identity    data/train_identity.csv    \
    --max_nodes   50000                      \
    --epochs      20                         \
    --model       sage                       \
    --full_batch
```

### 4. Launch Dashboard

```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501)

### 5. Generate Interactive PyVis Graph

```bash
python utils/visualization.py
```

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **AUC-ROC** | ~0.94 |
| Precision | ~0.82 |
| Recall | ~0.78 |
| F1-Score | ~0.80 |

> Results vary by training config. GPU + full dataset + 50 epochs recommended.

---

## 🧠 Model Details

### Graph Attention Network (GAT)

```
Input Features (transaction) → LazyLinear → 128d
  ↓ HeteroConv(GATConv, heads=4) per edge type
BatchNorm → ReLU → Dropout(0.3)
  ↓ HeteroConv(GATConv, heads=4)
BatchNorm → ReLU → Dropout(0.3)
  ↓ HeteroConv(GATConv, heads=1)
  ↓ Linear(128→64) → ReLU → Dropout → Linear(64→2)
Output: [P(Legitimate), P(Fraud)]
```

**Why GAT over GraphSAGE?**
- Attention weights reveal WHICH neighbors influence the prediction
- Fraudulent transactions have abnormal neighbor patterns → attention amplifies this
- Multi-head attention reduces variance from noisy/benign connections

### Handling Class Imbalance

```python
# ~3.5% fraud rate → severe imbalance
# Solution: Weighted Cross-Entropy
w0 = total / (2 * n_legitimate)   # ~0.52
w1 = total / (2 * n_fraud)        # ~14.3

loss = CrossEntropyLoss(weight=[w0, w1])
```

---

## 🔍 Explainability

### Per-Transaction Explanation (GNNExplainer)
```python
from models.explainability import explain_single_transaction
result = explain_single_transaction(model, data, tx_idx=42)
# Returns: fraud_prob, node_mask, edge_mask
```

### Global Feature Importance (Gradient × Input)
```python
from models.explainability import compute_gradient_feature_importance
importance = compute_gradient_feature_importance(model, data, mask, feature_names)
# Top features: TransactionAmt, card1, C13, D10, V258…
```

### Fraud Cluster Detection
```python
from models.explainability import detect_fraud_clusters
clusters = detect_fraud_clusters(G, fraud_probs, threshold=0.5)
# Returns list of suspicious node clusters (fraud rings)
```

---

## ☁️ Deployment

### Streamlit Cloud

```bash
git add . && git commit -m "Initial commit" && git push
# Go to share.streamlit.io → Connect repo → Set main file: app.py
# Note: Pre-compute artifacts locally and commit models/saved/ to repo
```

### Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

```bash
docker build -t fraud-gnn .
docker run -p 8501:8501 fraud-gnn
```

---

## 🏆 What Makes This Top 5%

1. **Heterogeneous graph** — most Kaggle solutions use flat features
2. **GNNExplainer** — interpretability is rare in competition notebooks
3. **Fraud ring detection** via community clustering — what banks actually care about
4. **Production architecture** — modular code, CLI args, model registry, dashboard
5. **Weighted loss + class analysis** — not SMOTE which distorts graph structure
6. **Entity node embeddings** — cards/devices learn their own behavior profile

### Bonus Improvements

| Improvement | Impact |
|-------------|--------|
| Add TransactionDT edges (temporal) | Captures velocity fraud |
| GraphTransformer instead of GAT | Better long-range dependencies |
| Contrastive learning (SimCLR on graphs) | Better fraud embeddings |
| Online learning (streaming graph updates) | Real-time fraud detection |
| Federated GNN (privacy-preserving) | Enterprise deployment |
| Feature store (Feast) | Production ML infra |
| MLflow experiment tracking | Reproducibility |
| K-fold cross-validation | More robust AUC estimate |

---

## 📝 Resume Bullets

```
- Architected a production-grade fraud detection system using Heterogeneous Graph
  Attention Networks (PyTorch Geometric) on 590K transactions, modeling card/device/
  email entity relationships across 5 node types and 4 edge types to detect fraud rings
  invisible to tabular ML — achieving AUC-ROC ~0.94.

- Engineered end-to-end ML pipeline: merged and preprocessed 590K+ transaction records,
  resolved 90%+ sparse features, handled 3.5% class imbalance via weighted cross-entropy
  loss, and deployed an interactive Streamlit dashboard with GNNExplainer interpretability
  and real-time fraud probability scoring.

- Applied graph-based explainability (GNNExplainer + Gradient×Input attribution) to surface
  top fraud predictors (TransactionAmt, card1 reuse, C-feature aggregates), enabling
  business-actionable fraud ring detection through Louvain community clustering on the
  transaction graph.
```

---

## 📚 References

- [IEEE-CIS Fraud Detection Dataset](https://www.kaggle.com/c/ieee-fraud-detection)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io)
- [GNNExplainer Paper](https://arxiv.org/abs/1903.03894)
- [Graph Attention Networks Paper](https://arxiv.org/abs/1710.10903)
- [Fraud Detection on Financial Networks with Multi-Scale GNN](https://arxiv.org/abs/2209.09612)

---

## 🙏 Acknowledgements
- [Kaggle IEEE-CIS](https://www.kaggle.com/c/ieee-fraud-detection) — For the dataset
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io) — For the GNN framework
- [Shields.io](https://shields.io) — For beautiful badges
- [Capsule Render](https://github.com/kyechan99/capsule-render) — For the animated banner
- [Streamlit](https://streamlit.io) — For the interactive dashboard

---

<div align="center">

**Anurag Kumar Upadhyay**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anurag-kumar-upadhyay-9a2105285/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AnuragKumar0429)
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:akupadhaya01@gmail.com)

<br/>

**If this project helped you, please ⭐ star the repo — it means the world!**

Made with ❤️ by **[Anurag Kumar Upadhyay](https://github.com/AnuragKumar0429)**

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=120&section=footer&animation=fadeIn" width="100%"/>

</div>
