import logging
import math
import torch

from latent_ablate.erasure import _is_meta
from latent_ablate.orthogonal_subspace import OrthogonalSubspaceProjector

logger = logging.getLogger("LatentAblate.SubspaceAlignment")


class SubspaceAlignmentAnalyzer:
    """Measures and aligns linear subspaces across layers or behavioral directions."""

    @staticmethod
    def compute_principal_angles(
        basis_a: torch.Tensor,
        basis_b: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Computes principal angles in radians between two subspaces spanned by rows of
        `basis_a` `[k_a, d]` and `basis_b` `[k_b, d]`.
        """
        if (
            basis_a is None
            or basis_b is None
            or _is_meta(basis_a)
            or _is_meta(basis_b)
            or basis_a.numel() == 0
            or basis_b.numel() == 0
        ):
            return torch.empty(0, dtype=torch.float32)

        ua = OrthogonalSubspaceProjector.gram_schmidt_orthonormalize(basis_a, eps=eps)
        ub = OrthogonalSubspaceProjector.gram_schmidt_orthonormalize(basis_b.to(ua.device), eps=eps)

        if ua.numel() == 0 or ub.numel() == 0 or ua.shape[-1] != ub.shape[-1]:
            return torch.empty(0, dtype=torch.float32, device=ua.device)

        # Cross-Gram matrix [m_a, m_b]; singular values equal cos(theta_i)
        cross = ua @ ub.T
        try:
            svals = torch.linalg.svdvals(cross)
        except RuntimeError:
            svals = torch.linalg.svdvals(cross.cpu()).to(ua.device)

        clamped = svals.clamp(0.0, 1.0)
        angles = torch.acos(clamped)
        return angles.contiguous()

    @classmethod
    def subspace_overlap_score(
        cls,
        basis_a: torch.Tensor,
        basis_b: torch.Tensor,
        eps: float = 1e-8,
    ) -> float:
        """
        Returns a normalized overlap score in `[0.0, 1.0]` based on squared cosines
        of principal angles (`1.0` = identical/aligned subspace, `0.0` = orthogonal).
        """
        angles = cls.compute_principal_angles(basis_a, basis_b, eps=eps)
        if angles.numel() == 0:
            return 0.0

        cos_sq = torch.cos(angles).pow(2)
        denom = max(1, min(
            int(OrthogonalSubspaceProjector.gram_schmidt_orthonormalize(basis_a, eps=eps).shape[0]),
            int(OrthogonalSubspaceProjector.gram_schmidt_orthonormalize(basis_b, eps=eps).shape[0]),
        ))
        score = float(cos_sq.sum().item()) / float(denom)
        if math.isnan(score) or math.isinf(score):
            return 0.0
        return max(0.0, min(1.0, round(score, 6)))

    @staticmethod
    def align_subspace_weights(
        source_weights: torch.Tensor,
        target_weights: torch.Tensor,
        blend_alpha: float = 0.5,
    ) -> torch.Tensor:
        """Blends `source_weights` toward `target_weights` for matching 2D weight shapes."""
        if (
            source_weights is None
            or target_weights is None
            or _is_meta(source_weights)
            or _is_meta(target_weights)
            or source_weights.numel() == 0
            or target_weights.numel() == 0
        ):
            return source_weights

        if source_weights.shape != target_weights.shape:
            logger.warning(
                f"Cannot align weights with mismatched shapes {tuple(source_weights.shape)} "
                f"vs {tuple(target_weights.shape)}."
            )
            return source_weights

        alpha = max(0.0, min(1.0, float(blend_alpha)))
        src_f = source_weights.detach().float()
        tgt_f = target_weights.detach().float().to(src_f.device)
        aligned = (1.0 - alpha) * src_f + alpha * tgt_f
        return aligned.to(device=source_weights.device, dtype=source_weights.dtype)
