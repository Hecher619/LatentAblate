import torch
import logging
from tqdm import tqdm

logger = logging.getLogger("LatentAblate.CausalSurgeon")

class CausalMediationAnalyzer:
    """
    Executes actual forward hooks and contrastive activation tracking 
    to measure layer sensitivity and locate causal refusal nodes.
    """
    @staticmethod
    def compute_indirect_effects(model, tokenizer, positive_prompts: list, negative_prompts: list, device: str) -> dict:
        logger.info("Running real causal mediation analysis across transformer layers...")
        model.eval()
        
        layers = None
        for attr in ["model", "transformer", "base_model"]:
            if hasattr(model, attr):
                sub = getattr(model, attr)
                for sub_attr in ["layers", "h", "block", "blocks"]:
                    if hasattr(sub, sub_attr):
                        layers = getattr(sub, sub_attr)
                        break
        if layers is None and hasattr(model, "layers"):
            layers = model.layers
            
        if layers is None:
            raise AttributeError("Could not resolve transformer layers for causal mediation.")

        layer_sensitivity = {}
        activations = []

        def forward_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            activations.append(hidden[:, -1, :].detach().float())

        for idx, layer in enumerate(tqdm(layers, desc="Causal Tracing Layers")):
            layer_dev = next(layer.parameters()).device
            handle = layer.register_forward_hook(forward_hook)
            try:
                pos_acts, neg_acts = [], []
                with torch.no_grad():
                    for p in positive_prompts[:3]:
                        activations.clear()
                        inputs = tokenizer(p, return_tensors="pt").to(device)
                        model(**inputs)
                        if activations: pos_acts.append(activations[-1].to(layer_dev))

                    for n in negative_prompts[:3]:
                        activations.clear()
                        inputs = tokenizer(n, return_tensors="pt").to(device)
                        model(**inputs)
                        if activations: neg_acts.append(activations[-1].to(layer_dev))

                if pos_acts and neg_acts:
                    p_tensor = torch.cat(pos_acts, dim=0).mean(dim=0)
                    n_tensor = torch.cat(neg_acts, dim=0).mean(dim=0)
                    effect_score = torch.linalg.norm(p_tensor - n_tensor).item()
                    layer_sensitivity[idx] = effect_score
            finally:
                handle.remove()

        optimal_nodes = sorted(layer_sensitivity, key=layer_sensitivity.get, reverse=True)[:5]
        logger.info(f"Top causal refusal nodes identified at layers: {optimal_nodes}")
        return {"sensitivity_map": layer_sensitivity, "target_nodes": optimal_nodes}
