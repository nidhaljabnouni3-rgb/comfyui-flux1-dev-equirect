import os

_TAG = "flux2-klein-models"

# VAE is saved outside the standard models/vae/ dir because that directory
# is symlinked to GCS FUSE in Cloud Run — writing there is slow/unreliable.
# We save to a local /tmp path and register it with folder_paths instead.
_VAE_LOCAL_DIR = "/tmp/flux2_klein_vae"

_MODELS = [
    {
        "label": "FLUX.2 [klein] 4B distilled diffusion model",
        "repo_id": "Comfy-Org/flux2-klein",
        "hf_path": "split_files/diffusion_models/flux-2-klein-4b.safetensors",
        "subdir": "diffusion_models",
        "filename": "flux-2-klein-4b.safetensors",
        "local_dir": None,  # resolved at runtime via folder_paths
    },
    {
        "label": "Qwen3 4B text encoder",
        "repo_id": "Comfy-Org/flux2-klein",
        "hf_path": "split_files/text_encoders/qwen_3_4b.safetensors",
        "subdir": "text_encoders",
        "filename": "qwen_3_4b.safetensors",
        "local_dir": None,
    },
    {
        "label": "FLUX.2 VAE",
        "repo_id": "Comfy-Org/flux2-dev",
        "hf_path": "split_files/vae/flux2-vae.safetensors",
        "subdir": "vae",
        "filename": "flux2-vae.safetensors",
        "local_dir": _VAE_LOCAL_DIR,  # bypass GCS symlink
    },
]


def _log(msg):
    print(f"[{_TAG}] {msg}", flush=True)


def _resolve_local_dir(model):
    if model["local_dir"] is not None:
        return model["local_dir"]
    try:
        import folder_paths
        paths = folder_paths.get_folder_paths(model["subdir"])
        if paths:
            return paths[0]
    except Exception:
        pass
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(root, "models", model["subdir"])


def _download_models():
    try:
        import requests
    except ImportError:
        _log("ERROR: requests not available, cannot download models")
        return

    token = os.environ.get("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    for model in _MODELS:
        local_dir = _resolve_local_dir(model)
        local_path = os.path.join(local_dir, model["filename"])

        if os.path.exists(local_path):
            _log(f"Already exists: {model['filename']}")
            continue

        os.makedirs(local_dir, exist_ok=True)
        url = f"https://huggingface.co/{model['repo_id']}/resolve/main/{model['hf_path']}"
        _log(f"Downloading {model['label']} ...")

        try:
            with requests.get(url, headers=headers, stream=True) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        f.write(chunk)
            size_gb = os.path.getsize(local_path) / 1e9
            _log(f"Saved: {local_path} ({size_gb:.1f} GB)")
        except Exception as e:
            _log(f"ERROR downloading {model['label']}: {e}")
            if os.path.exists(local_path):
                os.remove(local_path)

    _log("All models ready.")


def _register_vae_path():
    """Tell ComfyUI's VAELoader to also scan our local VAE dir."""
    try:
        import folder_paths
        folder_paths.add_model_folder_path("vae", _VAE_LOCAL_DIR)
    except Exception as e:
        _log(f"WARNING: could not register VAE path with folder_paths: {e}")


_download_models()
_register_vae_path()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
