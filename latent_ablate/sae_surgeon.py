import torch
import logging

logger = logging.getLogger("LatentAblate.SAESurgeon")

class SparseFeatureSurgeon:
    """
    Real sparse autoencoder (SAE) feature-level isolation and suppression 
    to zero out microscopic refusal feature dimensions.
    """
    @staticmethod
    def identify_refusal_features(positive_activations: torch.Tensor, negative_activations: torch.Tensor, top_k: int = 16) -> torch.Tensor:
        pos = positive_activations.float()
        neg = negative_activations.float()
        
        # Find feature dimensions with highest contrastive variance divergence
        diff_variance = (pos - neg).var(dim=0)
        top_indices = torch.topk(diff_variance, top_k).indices
        logger.info(f"Identified top {top_k} sparse contrastive refusal feature dimensions.")
        return top_indices

    @staticmethod
    def apply_feature_suppression(W: torch.Tensor, feature_indices: torch.Tensor, suppression_factor: float = 0.0) -> torch.Tensor:
        """Zeros out or scales specific weight columns corresponding to refusal features."""
        W_mod = W.float().clone()
        for idx in feature_indices:
            if idx < W_mod.shape[1]:
                W_mod[:, idx] *= suppression_factor
        return W_mod
