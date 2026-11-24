import logging
import torch

from latent_ablate.erasure import _is_meta

logger = logging.getLogger("LatentAblate.OrthogonalSubspace")


class OrthogonalSubspaceProjector:
    """Computes orthonormal subspace bases and removes orthogonal components from vectors/matrices."""

    @staticmethod
    def gram_schmidt_orthonormalize(
        basis_vectors: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Orthonormalizes a 2D tensor `[k, d]` via modified Gram-Schmidt with re-orthogonalization.
        Returns an orthonormal basis `[m, d]` where `m <= k` (linearly dependent/near-zero rows dropped).
        """
        if basis_vectors is None or _is_meta(basis_vectors) or basis_vectors.numel() == 0:
            return torch.empty(0, dtype=torch.float32)

        mat = basis_vectors.detach().float()
        if mat.ndim == 1:
            mat = mat.unsqueeze(0)
        if mat.ndim != 2 or mat.shape[0] == 0 or mat.shape[1] == 0:
            return torch.empty(0, dtype=torch.float32, device=mat.device)

        floor = max(1e-12, float(eps))
        ortho_rows: list[torch.Tensor] = []

        for row in mat:
            v = row.clone()
            for _ in range(2):
                for u in ortho_rows:
                    v = v - torch.dot(v, u) * u
            v_norm = float(v.norm().item())
            if v_norm > floor:
                ortho_rows.append((v / v_norm).contiguous())

        if not ortho_rows:
            return torch.empty(0, mat.shape[1], dtype=torch.float32, device=mat.device)

        return torch.stack(ortho_rows, dim=0).contiguous()

    @classmethod
    def project_out_orthogonal_subspace(
        cls,
        target: torch.Tensor,
        subspace_basis: torch.Tensor,
        scale: float = 1.0,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Projects `target` (1D vector or 2D row matrix) onto the orthogonal complement of
        `subspace_basis` (or scales the subspace component by `1.0 - scale`).
        """
        if target is None or _is_meta(target) or target.numel() == 0:
            return target
        if subspace_basis is None or _is_meta(subspace_basis) or subspace_basis.numel() == 0:
            return target

        t_f = target.detach().float()
        was_1d = t_f.ndim == 1
        if was_1d:
            t_f = t_f.unsqueeze(0)

        ortho = cls.gram_schmidt_orthonormalize(subspace_basis.to(t_f.device), eps=eps)
        if ortho.numel() == 0 or ortho.shape[-1] != t_f.shape[-1]:
            return target

        # t_f: [N, D], ortho: [M, D] -> coeffs: [N, M], proj: [N, D]
        coeffs = t_f @ ortho.T
        proj = coeffs @ ortho
        blend = max(0.0, min(1.0, float(scale)))
        result = t_f - blend * proj

        if was_1d:
            result = result.squeeze(0)
        return result.to(device=target.device, dtype=target.dtype)
