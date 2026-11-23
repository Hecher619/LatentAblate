import logging
from pathlib import Path

logger = logging.getLogger("LatentAblate.Publisher")

try:
    from huggingface_hub import HfApi, create_repo
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class HuggingFacePublisher:
    @staticmethod
    def publish_model(output_dir: str, repo_id: str, private: bool = True):
        """Pushes the liberated model directory to the Hugging Face Hub."""
        if not HF_AVAILABLE:
            raise ImportError("huggingface_hub is required for publishing. Run `pip install huggingface_hub`.")

        out_path = Path(output_dir).expanduser().resolve()
        if not out_path.exists():
            raise FileNotFoundError(f"Output model directory does not exist: {out_path}")

        logger.info(f"Connecting to Hugging Face Hub to publish repo: {repo_id}...")
        api = HfApi()

        try:
            create_repo(repo_id=repo_id, private=private, exist_ok=True)
            logger.info(f"Uploading files from {out_path} to {repo_id}...")

            api.upload_folder(
                folder_path=str(out_path),
                repo_id=repo_id,
                repo_type="model",
                commit_message="Upload liberated model via LatentAblate VELORA v2.1",
            )
            logger.info(f"Successfully published model to https://huggingface.co/{repo_id}")
        except Exception as e:
            logger.error(f"Failed to publish model to Hugging Face Hub: {e}")
            raise
