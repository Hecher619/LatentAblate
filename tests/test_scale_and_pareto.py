import json
import torch
import torch.nn as nn

from latent_ablate.arch import hidden_size, layer_index_from_ratio, resolve_layers
from latent_ablate.cli import build_parser
from latent_ablate.extractor import ActivationExtractor
from latent_ablate.prompts import PromptLoader
from latent_ablate.subspace import _winsorize
from latent_ablate.velora import VeloraEngine


class _DummyBlock(nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        self.self_attn = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return x + self.self_attn(x) + self.mlp(x)


class _DummyVLM(nn.Module):
    def __init__(self, n_layers: int = 4, dim: int = 16, vocab_size: int = 64):
        super().__init__()
        self.config = type("Cfg", (), {"text_config": type("TextCfg", (), {"hidden_size": dim, "vocab_size": vocab_size})()})()
        self.embed = nn.Embedding(vocab_size, dim)
        self.language_model = type("LM", (), {"layers": nn.ModuleList([_DummyBlock(dim) for _ in range(n_layers)])})()
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        h = self.embed(input_ids)
        for layer in self.language_model.layers:
            h = layer(h)
        logits = self.lm_head(h)
        return type("Out", (), {"logits": logits})()


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    def decode(self, token_ids, skip_special_tokens=True):
        vals = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
        chars = []
        for tok in vals:
            t = int(tok)
            if skip_special_tokens and t in (self.pad_token_id, self.eos_token_id):
                continue
            chars.append(chr(max(32, min(126, t))))
        return "".join(chars)
    chat_template = None
    padding_side = "right"

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=512):
        if isinstance(texts, str):
            texts = [texts]
        encoded = [[max(2, (ord(ch) % 50) + 2) for ch in t[:max_length]] or [2] for t in texts]
        max_len = max(len(seq) for seq in encoded)
        ids = []
        masks = []
        for seq in encoded:
            pad_len = max_len - len(seq)
            if self.padding_side == "left":
                ids.append([0] * pad_len + seq)
                masks.append([0] * pad_len + [1] * len(seq))
            else:
                ids.append(seq + [0] * pad_len)
                masks.append([1] * len(seq) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


def test_arch_vlm_layer_resolution_and_depth_ratio():
    model = _DummyVLM(n_layers=6, dim=16)
    layers = resolve_layers(model)
    assert len(layers) == 6
    assert hidden_size(model) == 16
    assert layer_index_from_ratio(6, 0.0) == 0
    assert layer_index_from_ratio(6, 1.0) == 5
    assert layer_index_from_ratio(32, 0.62) == 19


def test_prompt_loader_supports_json_jsonl_and_dedup(tmp_path):
    jsonl_file = tmp_path / "pairs.jsonl"
    rows = [
        {"positive": "Prompt A", "negative": "Benign A"},
        {"positive": "Prompt A", "negative": "Benign A"},  # duplicate
        {"prompt": "Prompt B", "label": "positive"},
        {"prompt": "Benign B", "label": "negative"},
    ]
    jsonl_file.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    pos, neg = PromptLoader.load_from_file(str(jsonl_file))
    assert pos == ["Prompt A", "Prompt B"]
    assert neg == ["Benign A", "Benign B"]


def test_extractor_caching_and_length_bucketing_order():
    model = _DummyVLM(n_layers=4, dim=16)
    tok = _DummyTokenizer()
    ext = ActivationExtractor(model, tok, device=torch.device("cpu"), batch_size=2, response_prefix="")

    prompts = [
        "short",
        "a much longer prompt string to exercise length bucketing",
        "medium length prompt",
        "short",  # duplicate should still return aligned row
    ]
    acts_first = ext.extract_hidden_states(prompts, layer_idx=1)
    assert acts_first.shape == (4, 16)
    assert torch.allclose(acts_first[0], acts_first[3], atol=1e-5)

    # Second call hits cache and preserves exact row ordering
    acts_cached = ext.extract_hidden_states(prompts, layer_idx=1)
    assert torch.allclose(acts_first, acts_cached, atol=1e-6)


def test_winsorize_handles_large_tensors_without_quantile_limit():
    big = torch.randn(600, 1024)  # > 500k elements
    out = _winsorize(big)
    assert out.shape == big.shape
    assert torch.isfinite(out).all()


def test_velora_subspace_extraction_and_projection():
    torch.manual_seed(0)
    pos = torch.randn(32, 16) + 1.5
    neg = torch.randn(32, 16)
    cap = torch.randn(8, 16)

    q_orth, weights = VeloraEngine.extract_velora_subspace(pos, neg, capability_basis=cap, max_rank=4)
    assert q_orth.shape[1] == 16
    assert q_orth.shape[0] == weights.shape[0]
    gram = q_orth @ q_orth.T
    assert torch.allclose(gram, torch.eye(q_orth.shape[0]), atol=1e-4)

    W = torch.randn(16, 16)
    W_mod = VeloraEngine.apply_velora_projection(W, q_orth, weights, alpha=1.0)
    assert W_mod.shape == W.shape
    assert not torch.allclose(W, W_mod)


def test_cli_zero_config_positional_model():
    parser = build_parser()
    args = parser.parse_args(["Qwen/Qwen2.5-1.5B-Instruct", "--preset", "gentle"])
    assert args.positional_model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert args.preset == "gentle"
