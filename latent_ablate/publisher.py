import logging
from pathlib import Path

logger = logging.getLogger("LatentAblate.Publisher")

try:
    from huggingface_hub import HfApi, create_repo
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class HuggingFacePublisher:
    """Publishes a saved model directory to the Hugging Face Hub."""

    @staticmethod
    def publish_model(
        output_dir: str,
        repo_id: str,
        private: bool = True,
        token: str | None = None,
        commit_message: str | None = None,
    ) -> str:
        if not HF_AVAILABLE:
            raise ImportError("`huggingface_hub` is required for publishing. Run `pip install huggingface_hub`.")

        clean_repo = (repo_id or "").strip()
        if not clean_repo or "/" not in clean_repo:
            raise ValueError(f"Invalid Hugging Face repo_id '{repo_id}'. Expected format 'namespace/repo-name'.")

        if not output_dir:
            raise ValueError("Output model directory must not be empty.")

        out_path = Path(output_dir).expanduser().resolve()
        if not out_path.exists() or not out_path.is_dir():
            raise FileNotFoundError(f"Output model directory does not exist: {out_path}")

        has_files = any(p.is_file() for p in out_path.rglob("*"))
        if not has_files:
            raise ValueError(f"Output model directory '{out_path}' contains no files to upload.")

        msg = commit_message or "Upload model via LatentAblate VELORA v2.2"
        logger.info(f"Publishing '{out_path}' to Hugging Face Hub repo '{clean_repo}' (private={private})...")

        api = HfApi(token=token) if token else HfApi()
        create_repo(repo_id=clean_repo, private=bool(private), exist_ok=True, token=token)
        api.upload_folder(
            folder_path=str(out_path),
            repo_id=clean_repo,
            repo_type="model",
            commit_message=msg,
            token=token,
        )

        url = f"https://huggingface.co/{clean_repo}"
        logger.info(f"Successfully published model to {url}")
        return url
