import os
import requests
import tarfile
from pathlib import Path

import trimesh

BUNNY_ZIP_URL = "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
CACHE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "cache")
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
INPUT_DIR = ASSETS_DIR / "input"
OUTPUT_DIR = ASSETS_DIR / "output"


def _asset_path(directory: Path, filename: str | os.PathLike[str]) -> Path:
    """Resolve a relative asset filename below the specified asset directory."""
    filename = Path(filename)
    if filename.is_absolute():
        raise ValueError(f"Asset path must be relative: {filename}")

    path = (directory / filename).resolve()
    try:
        path.relative_to(directory.resolve())
    except ValueError as exc:
        raise ValueError(f"Asset path must stay inside {directory}: {filename}") from exc
    return path


def load_mesh(filename: str | os.PathLike[str]) -> trimesh.Trimesh:
    """Load a mesh from ``assets/input``.

    ``filename`` may include subdirectories, but it must remain inside the input
    asset directory. GLB scenes are loaded as a single :class:`Trimesh`.
    """
    path = _asset_path(INPUT_DIR, filename)
    return trimesh.load_mesh(path, force="mesh")


def save_mesh(mesh: trimesh.Trimesh, filename: str | os.PathLike[str]) -> Path:
    """Save a mesh below ``assets/output`` and return the written path."""
    path = _asset_path(OUTPUT_DIR, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return path


def download_file(url, path):
    print(f"Downloading from {url} ...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    with open(path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Saved to {path}")


def extract_tar(path, extract_dir):
    print(f"Extracting {path} to {extract_dir}...")
    with tarfile.open(path) as tar:
        tar.extractall(extract_dir)
    print(f"Extracted to {extract_dir}")


def get_bunny() -> trimesh.Trimesh:
    BUNNY_PATH = os.path.join(CACHE_DIR, "bunny", "reconstruction", "bun_zipper.ply")
    if not os.path.exists(BUNNY_PATH):
        os.makedirs(CACHE_DIR, exist_ok=True)
        download_file(BUNNY_ZIP_URL, os.path.join(CACHE_DIR, "bunny.tar.gz"))
        extract_tar(os.path.join(CACHE_DIR, "bunny.tar.gz"), CACHE_DIR)
    return trimesh.load(BUNNY_PATH)
