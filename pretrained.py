"""Find a local folder or a Hugging Face repo that holds Mini's files."""

from pathlib import Path


def resolve_pretrained_folder(path_or_repo):
    """Return a directory with config.json. Download a Hub repo id first."""
    folder = Path(path_or_repo)
    if folder.is_dir():
        return folder
    from huggingface_hub import snapshot_download

    downloaded = snapshot_download(repo_id=str(path_or_repo))
    return Path(downloaded)
