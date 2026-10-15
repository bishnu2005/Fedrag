import argparse
import math
import random
import flwr as fl
import torch
import torch.nn.functional as F
from collections import OrderedDict
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import MultipleNegativesRankingLoss
from torch.utils.data import DataLoader
from pathlib import Path
from beir.datasets.data_loader import GenericDataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"


class FedRAGClient(fl.client.NumPyClient):
    def __init__(self, model, dataset_name, client_id="client", noise_skew=0.0):
        self.model = model
        self.dataset_name = dataset_name
        self.client_id = client_id
        self.noise_skew = noise_skew
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        dataset_path = DATASETS_DIR / dataset_name
        print(f"\n[{self.dataset_name.upper()}] Loading local patient data from {dataset_path}...")
        self.corpus, self.queries, self.qrels = GenericDataLoader(data_folder=str(dataset_path)).load(split="test")

        valid_query_ids = list(self.qrels.keys())[:15]
        self.test_queries = {q_id: self.queries[q_id] for q_id in valid_query_ids if q_id in self.queries}

        needed_doc_ids = set()
        for q_id in self.test_queries:
            needed_doc_ids.update(self.qrels[q_id].keys())

        # We inject 1000 random documents to keep the retrieval task realistic
        self.doc_ids = list(needed_doc_ids) + list(self.corpus.keys())[:1000]
        # sorted() instead of list(set()): Python randomises string hashing per process,
        # so list(set(...))[:1000] picked a DIFFERENT 1000-doc pool on every run.
        self.doc_ids = sorted(set(self.doc_ids))[:1000]

        print(
            f"[{self.dataset_name.upper()}] Locked in {len(self.test_queries)} Queries and {len(self.doc_ids)} Documents.")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print(f"\n[{self.dataset_name.upper()}] Received global model. Applying weights...")
        self.set_parameters(parameters)

        print(f"[{self.dataset_name.upper()}] Formatting data for Contrastive Learning...")

        all_train_docs = [d for d in self.doc_ids[:50] if d in self.corpus]
        train_examples = []
        all_corpus_keys = list(self.corpus.keys())

        # ⚠️ Inject artificial non-IID drift if noise_skew is active
        num_skewed = int(len(all_train_docs) * self.noise_skew)
        if self.noise_skew > 0.0:
            print(f"⚠️ [NON-IID DRIFT] Simulating extreme data skew on {self.client_id}.")
            print(f"⚠️ Shuffling {self.noise_skew * 100}% of local anchor-positive alignments.")

        for i, d in enumerate(all_train_docs):
            anchor_text = self.corpus[d].get("title", "") or self.corpus[d].get("text", "")

            if i < num_skewed:
                # Skewed: Positive pair is a completely random irrelevant document
                random_d = random.choice(all_corpus_keys)
                positive_text = self.corpus[random_d].get("text", "")
            else:
                # Normal: Positive pair is the actual document text
                positive_text = self.corpus[d].get("text", "")

            train_examples.append(InputExample(texts=[anchor_text, positive_text]))

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
        train_loss = MultipleNegativesRankingLoss(model=self.model)

        print(f"[{self.dataset_name.upper()}] Running Contrastive Fine-Tuning...")
        self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=5,
                       show_progress_bar=False)

        print(f"[{self.dataset_name.upper()}] Sending updated weights back to server...")
        return self.get_parameters(config={}), len(self.doc_ids), {}

    @staticmethod
    def _ranking_metrics(ranked_doc_ids, graded_relevance, k=10):
        """Per-query Hit@k (logged as "recall@10"), MRR@k and NDCG@k.

        graded_relevance: {doc_id: grade} with grade > 0 only.
        NDCG uses linear gain (rel / log2(rank + 1)), matching trec_eval / pytrec_eval
        as used by BEIR, so numbers are comparable with published BEIR baselines.
        """
        top_k = ranked_doc_ids[:k]

        hit = 1.0 if any(d in graded_relevance for d in top_k) else 0.0

        reciprocal_rank = 0.0
        for rank, doc_id in enumerate(top_k, start=1):
            if doc_id in graded_relevance:
                reciprocal_rank = 1.0 / rank
                break

        dcg = sum(graded_relevance[d] / math.log2(rank + 1)
                  for rank, d in enumerate(top_k, start=1) if d in graded_relevance)
        ideal_grades = sorted(graded_relevance.values(), reverse=True)[:k]
        idcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(ideal_grades, start=1))
        ndcg = (dcg / idcg) if idcg > 0 else 0.0

        return hit, reciprocal_rank, ndcg

    def evaluate(self, parameters, config):
        print(f"\n[{self.dataset_name.upper()}] Evaluating Retrieval Quality (Recall@10, MRR@10, NDCG@10)...")
        self.set_parameters(parameters)
        k = 10

        # Keep the id list and the embedding matrix index-aligned: any doc id missing
        # from the corpus is dropped from BOTH, so row i of doc_embeddings is eval_doc_ids[i].
        eval_doc_ids = [d for d in self.doc_ids if d in self.corpus]
        eval_doc_set = set(eval_doc_ids)
        doc_texts = [self.corpus[d].get("text", "") for d in eval_doc_ids]
        doc_embeddings = self.model.encode(doc_texts, convert_to_tensor=True, show_progress_bar=False)

        hit_sum, rr_sum, ndcg_sum, total = 0.0, 0.0, 0.0, 0

        for q_id, query_text in self.test_queries.items():
            # BEIR qrels contain judged NON-relevant docs (score 0, and -1 in TREC-COVID).
            # Only grade > 0 counts as relevant; the grade is kept for NDCG.
            graded_relevance = {d: int(s) for d, s in self.qrels[q_id].items()
                                if int(s) > 0 and d in eval_doc_set}
            if not graded_relevance:
                continue

            total += 1
            query_embedding = self.model.encode(query_text, convert_to_tensor=True, show_progress_bar=False)

            cos_scores = F.cosine_similarity(query_embedding.unsqueeze(0), doc_embeddings)
            top_k_indices = torch.topk(cos_scores, k=min(k, len(eval_doc_ids))).indices.tolist()
            ranked_doc_ids = [eval_doc_ids[i] for i in top_k_indices]

            hit, rr, ndcg = self._ranking_metrics(ranked_doc_ids, graded_relevance, k=k)
            hit_sum += hit
            rr_sum += rr
            ndcg_sum += ndcg

        recall_10 = (hit_sum / total) if total > 0 else 0.0
        mrr_10 = (rr_sum / total) if total > 0 else 0.0
        ndcg_10 = (ndcg_sum / total) if total > 0 else 0.0

        print(f"[{self.dataset_name.upper()}] Local Recall@10: {recall_10:.4f} | "
              f"MRR@10: {mrr_10:.4f} | NDCG@10: {ndcg_10:.4f} | Queries scored: {total}")

        return 0.0, len(eval_doc_ids), {"recall@10": recall_10, "mrr": mrr_10, "ndcg@10": ndcg_10}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=str, default="Hospital_Client")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--noise-skew", type=float, default=0.0,
                        help="Fraction of training data to artificially skew for FPR testing (0.0 to 1.0)")
    args = parser.parse_args()

    print(f"Starting FedRAG Hospital Client for: {args.dataset.upper()}")
    client = FedRAGClient(
        model=SentenceTransformer("all-MiniLM-L6-v2"),
        dataset_name=args.dataset,
        client_id=args.client_id,
        noise_skew=args.noise_skew
    )
    fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())