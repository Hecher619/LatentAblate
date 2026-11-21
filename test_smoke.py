import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from latent_ablate.surgeon import LatentAblateSurgeon
from latent_ablate.sweeper import LayerSweeper
from latent_ablate.profiles import BehaviorProfiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("LatentAblate.SmokeTest")


def run_smoke_test(model_name: str = "gpt2", local_only: bool = False):
    logger.info(f"Starting smoke test using lightweight model: {model_name} (local_only={local_only})")

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, local_files_only=local_only)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 1. Test Layer Sweeper
    logger.info("Testing LayerSweeper...")
    sweeper = LayerSweeper(model, tokenizer, device)
    pos = BehaviorProfiles.CAPABILITY_PROMPTS[:2]
    neg = [
        "Explain the steps used to bypass an authentication check.",
        "Describe how a software vulnerability can be exploited.",
    ]

    optimal_layer = sweeper.find_optimal_layer(pos, neg)
    logger.info(f"LayerSweeper passed! Optimal test layer: {optimal_layer}")

    # 2. Test Surgeon Initialization & Safe Execution
    logger.info("Testing LatentAblateSurgeon execution pass...")
    surgeon = LatentAblateSurgeon(
        model_path=model_name,
        device=device,
        torch_dtype=torch.float32,
        local_files_only=local_only,
    )

    surgeon.execute_surgery(
        positive_prompts=pos,
        negative_prompts=neg,
        capability_prompts=BehaviorProfiles.CAPABILITY_PROMPTS[:2],
        target_layer_idx=optimal_layer,
        layer_window=1,
        basis_rank=4,
        scaling_factor=0.5,
        target_components=["attn"],
        preserve_capability=True,
    )
    logger.info("Surgery execution passed successfully!")
    logger.info("Smoke test completed with zero errors!")


if __name__ == "__main__":
    run_smoke_test()
