import torch
import logging

logger = logging.getLogger("LatentAblate.SparseFeatures")


class SparseFeatureExtractor:
    @staticmethod
    def extract_sparse_basis(activations: torch.Tensor, l1_lambda: float = 1e-3, top_k: int = 8) -> torch.Tensor:
        """
        Learns an overcomplete, sparse feature basis from contrastive activations,
        isolating independent semantic axes for fine-grained suppression.
        """
        acts = activations.float()
        mean = acts.mean(dim=0, keepdim=True)
        centered = acts - mean

        _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
        k = max(1, min(int(top_k), Vh.shape[0]))
        weights = Vh[:k]
        threshold = float(l1_lambda) * weights.abs().max(dim=1, keepdim=True)[0]
        sparse_basis = torch.sign(weights) * torch.clamp(weights.abs() - threshold, min=0.0)
        sparse_basis = torch.nn.functional.normalize(sparse_basis, p=2, dim=1)
        logger.info(f"Extracted {k} sparse interpretable feature axes successfully.")
        return sparse_basis
