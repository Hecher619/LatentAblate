import torch
import logging
from latent_ablate.arch import hidden_size, resolve_layers

logger = logging.getLogger("LatentAblate.Diagnostics")

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ModelDiagnostics:
    @staticmethod
    def audit_model(model, tokenizer) -> dict:
        """Performs a deep structural pre-flight check on the model before latent surgery."""
        logger.info("Running pre-flight structural model audit...")
        
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        h_size = hidden_size(model)
        
        try:
            layers = resolve_layers(model)
            num_layers = len(layers)
        except Exception as e:
            logger.error(f"Failed to resolve transformer layers: {e}")
            num_layers = -1

        is_quantized = getattr(model, "is_quantized", False) or "bitsandbytes" in str(type(model)).lower()
        model_type = getattr(model.config, "model_type", "unknown")

        audit_info = {
            "model_type": model_type,
            "device": str(device),
            "dtype": str(dtype),
            "hidden_size": h_size,
            "total_layers": num_layers,
            "is_quantized": is_quantized
        }

        if RICH_AVAILABLE:
            table = Table(title="LatentAblate v5.0 - Pre-Flight Model Audit")
            table.add_column("Property", style="cyan", no_wrap=True)
            table.add_column("Detected Value", style="green")
            for k, v in audit_info.items():
                table.add_row(k.replace("_", " ").title(), str(v))
            console.print(table)
        else:
            logger.info(f"Audit Results: {audit_info}")

        return audit_info
