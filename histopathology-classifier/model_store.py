"""Fetch trained models from the GitHub release when they are not local."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


RELEASE_URL = "https://api.github.com/repos/adi-issa/AIML_P2/releases/tags/models-v1"
MODEL_SHA256 = {
    "resnet50v2_Tuning.keras": "515589e9330ef4ad8250724dd9574da437f0b131a5c1cb06576b9987139e14e3",
    "densenet169_Tuning.keras": "3a20ca66c183cd11b33a4f4eeb190c33a707fc088365ee37ca8d852792642f55",
}


def _request(url: str, token: str | None, accept: str) -> Request:
    headers = {
        "Accept": accept,
        "User-Agent": "histopathology-classifier",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return Request(url, headers=headers)


def ensure_model_file(filename: str, local_dir: Path, token: str | None) -> Path:
    """Return a local model, downloading and verifying it if necessary."""
    if filename not in MODEL_SHA256:
        raise ValueError(f"Unknown trained model: {filename}")

    local_path = local_dir / filename
    if local_path.is_file():
        return local_path

    cache_dir = Path.home() / ".cache" / "histopathology-classifier" / "models"
    cache_path = cache_dir / filename
    if cache_path.is_file():
        return cache_path

    try:
        with urlopen(_request(RELEASE_URL, token, "application/vnd.github+json"), timeout=30) as response:
            release = json.load(response)
        asset = next((item for item in release["assets"] if item["name"] == filename), None)
        if asset is None:
            raise RuntimeError(f"{filename} is missing from the models-v1 GitHub release.")

        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".download", delete=False) as temporary:
                temporary_path = Path(temporary.name)
                digest = hashlib.sha256()
                with urlopen(_request(asset["url"], token, "application/octet-stream"), timeout=120) as response:
                    while chunk := response.read(1024 * 1024):
                        temporary.write(chunk)
                        digest.update(chunk)
            if digest.hexdigest() != MODEL_SHA256[filename]:
                raise RuntimeError(f"The downloaded {filename} failed its SHA-256 check.")
            os.replace(temporary_path, cache_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    except HTTPError as exc:
        raise RuntimeError(
            f"GitHub returned HTTP {exc.code} while fetching {filename}. "
            "If the repository is private, set GITHUB_TOKEN in Streamlit app secrets."
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Could not connect to GitHub to fetch {filename}.") from exc
    return cache_path
