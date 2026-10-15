import torch
import flwr as fl
import numpy as np
from fedrag.client.flower_client import FedRAGClient


class KrumBypassAttacker(FedRAGClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, parameters, config):
        # 1. Perform standard contrastive training to get the true, geometrically "normal" update
        true_results = super().fit(parameters, config)
        true_new_weights = true_results[0]

        original_tensors = [torch.tensor(p) for p in parameters]
        trained_tensors = [torch.tensor(p) for p in true_new_weights]

        # 2. Extract the raw parameter updates (Gradients)
        updates = [train_w - orig_w for train_w, orig_w in zip(trained_tensors, original_tensors)]

        # 3. Calculate the true L2 norm. We MUST match this exactly to bypass Krum/Median.
        total_l2_norm = torch.sqrt(sum(torch.sum(u ** 2) for u in updates))

        adversarial_updates = []
        for u, name in zip(updates, self.model.state_dict().keys()):
            # 4. Target ONLY the semantic projection layers for scrambling
            if "attention.self.value" in name or "attention.output.dense" in name:
                # Permute/scramble the matrix along its first dimension to destroy semantic mapping
                if u.dim() > 1:
                    scrambled = u[torch.randperm(u.size(0))]
                else:
                    scrambled = u
                adversarial_updates.append(scrambled)
            else:
                # Leave base structural layers completely untouched
                adversarial_updates.append(u)

        # 5. Re-normalize the poisoned update to guarantee the exact original L2 distance
        adv_l2_norm = torch.sqrt(sum(torch.sum(au ** 2) for au in adversarial_updates))
        scaling_factor = total_l2_norm / (adv_l2_norm + 1e-8)

        final_adversarial_weights = []
        for orig_w, adv_u in zip(original_tensors, adversarial_updates):
            poisoned_w = orig_w + (adv_u * scaling_factor)
            final_adversarial_weights.append(poisoned_w.numpy())

        print("\n🥷 [BYZANTINE ATTACK] Krum-Bypass active.")
        print(
            f"Original L2 Norm: {total_l2_norm.item():.4f} | Poisoned L2 Norm: {(adv_l2_norm * scaling_factor).item():.4f}")
        print("Semantic projection layers scrambled. Bypassing geometric filters...")

        return final_adversarial_weights, true_results[1], true_results[2]


# To run this client:
if __name__ == "__main__":
    import argparse
    from sentence_transformers import SentenceTransformer  # Added import

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    # Initialize the attacker WITH the required model argument
    client = KrumBypassAttacker(
        model=SentenceTransformer("all-MiniLM-L6-v2"),
        dataset_name=args.dataset,
        client_id="byzantine_scifact"
    )
    fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())