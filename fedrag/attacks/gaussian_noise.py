import argparse
import flwr as fl
import torch
import numpy as np
from collections import OrderedDict
from sentence_transformers import SentenceTransformer
from pathlib import Path
from beir.datasets.data_loader import GenericDataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"


class GaussianNoiseClient(fl.client.NumPyClient):
    def __init__(self, model, dataset_name, noise_std=0.5):
        self.model = model
        self.dataset_name = dataset_name
        self.noise_std = noise_std
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        dataset_path = DATASETS_DIR / dataset_name
        print(f"\n[GAUSSIAN NOISE ATTACKER - {self.dataset_name.upper()}] Loading dataset...")
        self.corpus, _, _ = GenericDataLoader(data_folder=str(dataset_path)).load(split="test")
        self.doc_ids = list(self.corpus.keys())[:100]

    def get_parameters(self, config):
        """Injects high-variance Gaussian noise into the model weights."""
        print(f"[ATTACKER] Injecting Gaussian Noise (std={self.noise_std}) into weight tensors...")
        corrupted_params = []
        for _, val in self.model.state_dict().items():
            param_np = val.cpu().numpy()
            noise = np.random.normal(loc=0.0, scale=self.noise_std, size=param_np.shape)
            corrupted_params.append(param_np + noise)
        return corrupted_params

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print(f"\n[⚡ ATTACKER] Received global model. Corrupting updates with Gaussian noise...")
        self.set_parameters(parameters)
        return self.get_parameters(config={}), len(self.doc_ids), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, len(self.doc_ids), {"recall@10": 0.0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedRAG Gaussian Noise Attacker")
    parser.add_argument("--dataset", type=str, default="scifact", help="Dataset name")
    parser.add_argument("--std", type=float, default=0.5, help="Standard deviation of noise")
    args = parser.parse_args()

    print(f"Starting FedRAG Gaussian Noise Attacker on: {args.dataset.upper()}")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=GaussianNoiseClient(model, args.dataset, noise_std=args.std).to_client()
    )