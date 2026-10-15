Here is the exact content you need. Just copy everything inside the block below, create a new file in your GitHub repository named **`README.md`**, and paste it in.

```markdown
# FedRAG: A Recall-Driven Byzantine Defense for Federated Retrieval Models

This repository contains the official implementation of **FedRAG**, a privacy-preserving federated embedding architecture designed specifically for dense Information Retrieval (IR) systems. 

FedRAG secures federated representation learning by replacing traditional proximity-based geometric filters (e.g., Krum, Coordinate-wise Median) with a task-grounded Evaluation Gate. By evaluating intercepted client updates against a held-out validation query using Recall@10, FedRAG triggers an exponential Soft Trust Decay mechanism to systematically isolate and block Byzantine attackers executing stealthy semantic space scrambling, sign-flipping, or Gaussian noise attacks.

## Architecture Overview

The FedRAG pipeline operates without generative LLMs and strictly outputs cryptographic document pointers and cosine relevance scores via an in-memory vector index (Qdrant). The federated framework is structured across six stages:

1. **Local Contrastive Fine-Tuning:** Honest clients fine-tune a local Sentence-Transformer (`all-MiniLM-L6-v2`) over private corpora via Multiple Negatives Ranking Loss (MNRL).
2. **Differential Privacy Shield:** Gradient updates undergo bounded L2 norm clipping to restrict leakage.
3. **Server Intercept:** The central server intercepts candidate updates before model aggregation.
4. **Evaluation Gate (Recall@10 Validation):** Updates are applied to an isolated temporary global model and evaluated on a held-out query validation set.
5. **Soft Trust Decay:** Malicious updates that degrade retrieval performance trigger an exponential decay in the client's persistent trust score (decay constant $\gamma = 0.5$).
6. **Safe Aggregation:** Parameters are aggregated using normalized trust weights across active clients, dynamically excluding blocked nodes.

## Repository Structure

```text
├── client/
│   ├── client.py          # Flower client initialization and gRPC communication
│   ├── local_train.py     # MNRL contrastive fine-tuning logic
│   └── data_loader.py     # BEIR dataset ingestion and partitioning
├── server/
│   ├── server.py          # Flower server orchestration
│   ├── strategy.py        # Custom Trust-Weighted Aggregation logic
│   └── validator.py       # Recall@10 Evaluation Gate & Soft Trust Decay mechanics
├── utils/
│   ├── metrics.py         # Retrieval evaluation (Recall@10, MRR@10, NDCG@10)
│   └── attacks.py         # Byzantine attack simulations (Sign-flip, Gaussian)
├── requirements.txt       # Project dependencies
└── README.md

```

## Prerequisites

The framework requires Python 3.9+ and an NVIDIA GPU for efficient contrastive fine-tuning.

```bash
git clone [https://github.com/bishnu2005/Fedrag.git](https://github.com/bishnu2005/Fedrag.git)
cd Fedrag
pip install -r requirements.txt

```

Core dependencies:

* `flwr` (Flower Federated Learning Framework)
* `sentence-transformers`
* `torch` (PyTorch)
* `qdrant-client`
* `beir`

## Datasets

The experimental evaluation utilizes three distinct, non-IID corpora from the BEIR benchmark to simulate highly regulated, heterogeneous institutional environments:

* **TREC-COVID:** Biomedical/virology corpus (simulating an infectious disease research center).
* **NFCorpus:** Medical nutrition corpus (simulating a clinical dietary institution).
* **SciFact:** Scientific claim verification corpus (utilized by the Byzantine client to execute targeted gradient corruption).

## Usage

**1. Initialize the FedRAG Server**
Start the central aggregation server with the custom Soft Trust Decay strategy.

```bash
python server/server.py --rounds 3 --threshold 0.25

```

**2. Launch Honest Clients**
In separate terminal instances, launch the honest clients attached to their respective local datasets.

```bash
python client/client.py --dataset trec-covid --cid 1
python client/client.py --dataset nfcorpus --cid 2

```

**3. Launch Byzantine Client**
Simulate an adversarial client executing a targeted sign-flipping attack ($\eta = 5.0$) or Gaussian noise attack ($\sigma = 0.5$).

```bash
python client/client.py --dataset scifact --cid 3 --attack sign_flip --eta 5.0

```

## Empirical Results

Our evaluations demonstrate that while standard FedAvg and FedMedian fail to maintain semantic alignment under targeted poisoning (yielding a degraded Recall@10 of 0.426), FedRAG successfully detects and suppresses malicious updates. The Soft Trust Decay framework reaches a high network utility (Recall@10 of 0.948) while perfectly preserving non-IID honest client utility with a 0.0% False Positive Rate.

## Citation

If you utilize this framework or code in your research, please cite the accompanying paper:

```bibtex
@inproceedings{mangaraj2026fedrag,
  title={Degrade to Decay: A Recall-Driven Byzantine Defense for Federated Retrieval Models},
  author={Mangaraj, Bishnu Prasad and Pandey, Tanishq and S., Gayathri},
  booktitle={IEEE},
  year={2026}
}

```

```

```