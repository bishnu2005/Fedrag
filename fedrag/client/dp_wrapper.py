import torch
from opacus import PrivacyEngine


class DifferentialPrivacyShield:
    def __init__(self, model, optimizer, data_loader, noise_multiplier=1.0, max_grad_norm=1.0):
        """
        Wraps the local SentenceTransformer training loop with Opacus to inject DP noise.
        """
        print("[🔒 DP SHIELD] Initializing Opacus Differential Privacy Engine...")
        self.privacy_engine = PrivacyEngine()

        # Opacus intercepts the model, optimizer, and dataloader to inject noise dynamically
        self.model, self.optimizer, self.data_loader = self.privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )
        print(f"[🔒 DP SHIELD] Model wrapped. Noise Multiplier: {noise_multiplier}, Max Grad Norm: {max_grad_norm}")

    def get_privacy_budget(self, target_delta=1e-5):
        """Calculates the epsilon privacy budget consumed during the local training phase."""
        epsilon = self.privacy_engine.get_epsilon(target_delta)
        print(f"[🔒 DP SHIELD] Privacy Budget Expended: ε = {epsilon:.2f}, δ = {target_delta}")
        return epsilon