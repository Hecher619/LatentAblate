import logging
import torch

from latent_ablate.arch import REFUSAL_MARKERS, format_chat_prompt
from latent_ablate.profiles import BehaviorProfiles
from latent_ablate.reaction import classify_response
from latent_ablate.validator import ModelValidator

logger = logging.getLogger("LatentAblate.Benchmark")


def run_mini_benchmark(
    model,
    tokenizer,
    device,
    over_refusal_prompts: list[str] | None = None,
    capability_prompts: list[str] | None = None,
    response_prefix: str = "",
) -> dict:
    """
    Runs a fast mini-benchmark checking:
    1) Benign compliance / false-refusal rate on tricky benign prompts (XSTest-style)
    2) Perplexity on capability preservation prompts
    """
    benign_set = over_refusal_prompts or BehaviorProfiles.OVER_REFUSAL_PROMPTS
    cap_set = capability_prompts or BehaviorProfiles.CAPABILITY_PROMPTS

    compliant = 0
    false_refusals = 0
    model.eval()

    for prompt in benign_set:
        formatted = format_chat_prompt(tokenizer, prompt, response_prefix=response_prefix)
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        in_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        reply = tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()
        assessment = classify_response(reply, REFUSAL_MARKERS)
        if assessment.is_refusal or assessment.is_moralizing:
            false_refusals += 1
        else:
            compliant += 1

    denom = max(1, len(benign_set))
    benign_compliance_pct = round((compliant / denom) * 100.0, 1)
    false_refusal_pct = round((false_refusals / denom) * 100.0, 1)
    ppl = (
        round(float(ModelValidator.calculate_perplexity(model, tokenizer, cap_set, device)), 2)
        if cap_set
        else 0.0
    )

    summary = {
        "benign_compliance_pct": benign_compliance_pct,
        "false_refusal_pct": false_refusal_pct,
        "capability_perplexity": ppl,
        "num_benign_prompts": len(benign_set),
    }
    logger.info(
        f"Mini-Benchmark -> Benign Compliance: {benign_compliance_pct:.1f}% "
        f"(False Refusals: {false_refusal_pct:.1f}%) | Capability PPL: {ppl:.2f}"
    )
    return summary
