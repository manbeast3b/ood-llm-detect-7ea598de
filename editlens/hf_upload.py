"""
Shared helper: push a produced model/adapter + its model card to the HF Hub so
experiment results survive the run box being torn down.

The namespace (account) is resolved at runtime from the configured HF token via
whoami(), and every "{{NS}}" placeholder in the card is rewritten to it — so the
cross-links between the three sibling models are always correct regardless of
which account's token is set. Failures are logged but never crash the run.
"""
import os
import traceback


def resolve_namespace():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        return None
    try:
        from huggingface_hub import HfApi
        return HfApi(token=token).whoami().get("name")
    except Exception:
        return None


def push_to_hub(local_dir, repo_suffix, card_text=None, private=False):
    """Upload everything in local_dir to <whoami>/<repo_suffix> as a model repo.

    `card_text` may contain "{{NS}}" placeholders; they are rewritten to the
    resolved namespace before writing README.md.
    Returns the repo URL, or None on failure/skip.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("[hf_upload] no HF token in env; skipping upload")
        return None
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        ns = api.whoami().get("name")
        repo_id = f"{ns}/{repo_suffix}"
        api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        if card_text:
            card_text = card_text.replace("{{NS}}", ns)
            with open(os.path.join(local_dir, "README.md"), "w") as f:
                f.write(card_text)
        api.upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model")
        url = f"https://huggingface.co/{repo_id}"
        print(f"[hf_upload] pushed -> {url}")
        return url
    except Exception:
        print("[hf_upload] upload FAILED (non-fatal):")
        traceback.print_exc()
        return None
