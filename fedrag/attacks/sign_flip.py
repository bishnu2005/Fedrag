import flwr as fl
import torch
from collections import OrderedDict
from sentence_transformers import SentenceTransformer
from pathlib import Path
from beir.datasets.data_loader import GenericDataLoader

# Dynamic Pathing
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"


class MaliciousFedRAGClient(fl.client.NumPyClient):
    def __init__(self, model, dataset_name):
        self.model = model
        self.dataset_name = dataset_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        dataset_path = DATASETS_DIR / dataset_name
        print(f"\n[MALICIOUS CLIENT - {self.dataset_name.upper()}] Loading data...")
        self.corpus, _, _ = GenericDataLoader(data_folder=str(dataset_path)).load(split="test")
        self.doc_ids = list(self.corpus.keys())[:100]

    def get_parameters(self, config):
        """THE ATTACK: Hijacks the weights and flips their mathematical signs to destroy the global model."""
        print(f"[ MALICIOUS CLIENT]️ INITIATING SIGN-FLIP ATTACK! Poisoning weights...")
        # We multiply the weights by -5.0 to violently corrupt the embedding space directionality
        return [(val.cpu().numpy() * -5.0) for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print(f"\n[ MALICIOUS CLIENT]  Received global model.")
        self.set_parameters(parameters)
        print(f"[ MALICIOUS CLIENT]  Pretending to train normally on {len(self.doc_ids)} docs...")
        # Returns the poisoned parameters back to the server
        return self.get_parameters(config={}), len(self.doc_ids), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, len(self.doc_ids), {"recall@10": 0.0}


if __name__ == "__main__":
    print("Starting FedRAG Malicious Client (Sign-Flip Attack)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # We hardcode this attacker to use the SciFact dummy dataset
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=MaliciousFedRAGClient(model, "scifact").to_client()
    )