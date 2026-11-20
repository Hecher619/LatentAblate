import logging
import torch
from latent_ablate.profiles import BehaviorProfiles
from latent_ablate.publisher import HuggingFacePublisher
from latent_ablate.reporter import SurgeryReporter
from latent_ablate.surgeon import LatentAblateSurgeon
from latent_ablate.validator import ModelValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("LatentAblate.Interactive")


def main():
    print("=" * 65)
    print("LatentAblate: Precision Model Surgery Suite")
    print("=" * 65)

    model_path = input("Enter Hugging Face model path or local directory [Qwen/Qwen2.5-1.5B-Instruct]: ").strip()
    if not model_path:
        model_path = "Qwen/Qwen2.5-1.5B-Instruct"

    output_dir = input("Enter output directory for liberated model [./output/liberated-model]: ").strip()
    if not output_dir:
        output_dir = "./output/liberated-model"

    target_layer = int(input("Enter target layer index for activation extraction [14]: ") or "14")
    rank = int(input("Enter SVD subspace rank (k) [8]: ") or "8")
    scaling = float(input("Enter orthogonal projection scaling factor (alpha) [0.70]: ") or "0.70")

    reporter = SurgeryReporter(output_dir)
    reporter.log_step("initialization", {
        "model_path": model_path,
        "target_layer_idx": target_layer,
        "basis_rank": rank,
        "scaling_factor": scaling,
    })

    profile = BehaviorProfiles.get_profile("safety_uncensored")
    positive_prompts = profile["positive"][:12]
    negative_prompts = profile["negative"][:12]
    capability_prompts = BehaviorProfiles.CAPABILITY_PROMPTS

    print("\nInitializing LatentAblate Surgeon...")
    surgeon = LatentAblateSurgeon(model_path=model_path, device="auto", torch_dtype=torch.float16)

    print("\nExecuting precision subspace orthogonalization & norm calibration...")
    surgeon.execute_surgery(
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        capability_prompts=capability_prompts,
        target_layer_idx=target_layer,
        layer_window=2,
        basis_rank=rank,
        scaling_factor=scaling,
    )
    reporter.log_step("surgery_executed", {"status": "success", "layers_modified_window": "target+-2"})

    print("\nRunning post-surgery validation report...")
    validation_results = ModelValidator.verify_surgery(
        surgeon.model,
        surgeon.tokenizer,
        surgeon.device,
        baseline_texts=capability_prompts,
    )
    reporter.log_step("validation", {"results": validation_results})

    surgeon.save_model(output_dir)
    reporter.save_report()
    print(f"\nModel successfully saved to {output_dir} along with surgery_report.json!")

    publish_choice = input("\nPublish this model to Hugging Face Hub? (y/n): ").strip().lower()
    if publish_choice == "y":
        repo_id = input("Enter HF repo ID (e.g., username/liberated-model-32b): ").strip()
        is_private = input("Make repository private? (y/n) [n]: ").strip().lower() == "y"
        if repo_id:
            HuggingFacePublisher.publish_model(output_dir, repo_id, private=is_private)


if __name__ == "__main__":
    main()
