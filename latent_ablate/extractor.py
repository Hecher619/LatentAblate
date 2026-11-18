import torch
import logging
from tqdm import tqdm

logger = logging.getLogger("LatentAblate.Extractor")

class ActivationExtractor:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _get_layers(self):
        for attr in ["model", "transformer", "base_model"]:
            if hasattr(self.model, attr):
                submodule = getattr(self.model, attr)
                for sub_attr in ["layers", "h", "block", "blocks"]:
                    if hasattr(submodule, sub_attr):
                        return getattr(submodule, sub_attr)
        if hasattr(self.model, "layers"):
            return self.model.layers
        raise AttributeError("Could not resolve transformer layers for activation extraction.")

    def extract_hidden_states(self, prompts: list, layer_idx: int) -> torch.Tensor:
        activations = []

        def forward_hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            activations.append(hidden_states[:, -1, :].detach().float())

        layers = self._get_layers()
        if not (0 <= layer_idx < len(layers)):
            raise IndexError(f"Layer index {layer_idx} out of range for model with {len(layers)} layers.")

        target_layer = layers[layer_idx]
        handle = target_layer.register_forward_hook(forward_hook)

        try:
            with torch.no_grad():
                for prompt in tqdm(prompts, desc=f"Extracting activations (Layer {layer_idx})"):
                    # Apply native chat template if supported by the tokenizer
                    if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
                        messages = [{"role": "user", "content": prompt}]
                        formatted_prompt = self.tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    else:
                        formatted_prompt = prompt

                    inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
                    self.model(**inputs)
        finally:
            handle.remove()

        if not activations:
            raise ValueError(f"Failed to capture activations at layer {layer_idx}.")

        return torch.cat(activations, dim=0)
