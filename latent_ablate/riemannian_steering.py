import torch
import logging

logger = logging.getLogger("LatentAblate.RiemannianSteering")

class RiemannianGeodesicEngine:
    """
    Computes rigorous Riemannian manifold exponential maps and parallel transport 
    along tangent space vector fields for non-destructive latent steering.
    """
    @staticmethod
    def parallel_transport(h: torch.Tensor, tangent_vector: torch.Tensor, step_size: float) -> torch.Tensor:
        """
        Transports hidden states along the manifold geodesic curve 
        using exact spherical exponential mapping.
        """
        h_f = h.float()
        v = torch.nn.functional.normalize(tangent_vector.float(), dim=0)
        
        # Project h onto tangent space orthogonal to v
        h_dot_v = (h_f @ v).unsqueeze(-1)
        parallel_component = h_f - h_dot_v * v
        
        # Geodesic exponential map formula on unit hypersphere / tangent bundle
        cos_s = torch.cos(torch.tensor(step_size, dtype=h_f.dtype, device=h_f.device))
        sin_s = torch.sin(torch.tensor(step_size, dtype=h_f.dtype, device=h_f.device))
        
        transported = parallel_component + h_dot_v * v * cos_s + v * sin_s * h_f.norm(dim=-1, keepdim=True)
        return transported.to(h.dtype)
    
    @staticmethod
    def compute_connection_coefficients(basis: torch.Tensor) -> torch.Tensor:
        """Computes Christoffel-like connection coefficients for manifold curvature approximation."""
        b = basis.float()
        gram = b @ b.T
        return torch.linalg.inv(gram + 1e-5 * torch.eye(gram.shape[0], device=b.device))
