import os

_TAG = "flux1-dev-equirect"

# Dirs symlinked to GCS FUSE in Cloud Run (writes are slow/unreliable).
# Small models (VAE, LoRA) go to /tmp.
# Large diffusion models go to models/diffusion_models/ which is symlinked
# to GCS FUSE by the entrypoint — persists across cold starts, no RAM usage.
_VAE_LOCAL_DIR = "/tmp/flux1dev_equirect_vae"
_LORA_LOCAL_DIR = "/tmp/flux1dev_equirect_loras"
_UPSCALE_LOCAL_DIR = "/tmp/flux1dev_equirect_upscale"

# Entrypoint symlinks models/ subdirs → /gcs/comfyui/models/ (GCS FUSE).
# This avoids filling the writable layer (RAM-backed) with large models.
_COMFYUI_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DIFFUSION_FUSE_DIR = os.path.join(_COMFYUI_DIR, "models", "diffusion_models")
_TEXT_ENC_FUSE_DIR = os.path.join(_COMFYUI_DIR, "models", "text_encoders")

_MODELS = [
    {
        "label": "FLUX.1-dev diffusion model (fp8)",
        "repo_id": "Kijai/flux-fp8",
        "hf_path": "flux1-dev-fp8.safetensors",
        "subdir": "diffusion_models",
        "filename": "flux1-dev-fp8.safetensors",
        "local_dir": None,  # local tmpfs (models/unet/) — efficient mmap, no RAM doubling
    },
    {
        "label": "FLUX.1-Fill-dev diffusion model (fp8)",
        "repo_id": "1038lab/FLUX.1-Fill-dev_fp8",
        "hf_path": "FLUX.1-Fill-dev_fp8.safetensors",
        "subdir": "diffusion_models",
        "filename": "FLUX.1-Fill-dev_fp8.safetensors",
        "local_dir": _DIFFUSION_FUSE_DIR,  # GCS FUSE via entrypoint symlink
    },
    {
        "label": "CLIP-L text encoder",
        "repo_id": "comfyanonymous/flux_text_encoders",
        "hf_path": "clip_l.safetensors",
        "subdir": "text_encoders",
        "filename": "clip_l.safetensors",
        "local_dir": _TEXT_ENC_FUSE_DIR,  # GCS FUSE via entrypoint symlink
    },
    {
        "label": "T5-XXL text encoder (fp8)",
        "repo_id": "comfyanonymous/flux_text_encoders",
        "hf_path": "t5xxl_fp8_e4m3fn.safetensors",
        "subdir": "text_encoders",
        "filename": "t5xxl_fp8_e4m3fn.safetensors",
        "local_dir": _TEXT_ENC_FUSE_DIR,  # GCS FUSE via entrypoint symlink
    },
    {
        "label": "FLUX.1 VAE (ae)",
        "repo_id": "ffxvs/vae-flux",
        "hf_path": "ae.safetensors",
        "subdir": "vae",
        "filename": "ae.safetensors",
        "local_dir": _VAE_LOCAL_DIR,  # bypass GCS symlink
    },
    {
        "label": "4x-UltraSharp upscaler",
        "repo_id": "Kim2091/UltraSharp",
        "hf_path": "4x-UltraSharp.pth",
        "subdir": "upscale_models",
        "filename": "4x-UltraSharp.pth",
        "local_dir": _UPSCALE_LOCAL_DIR,  # bypass GCS symlink
    },
    {
        "label": "Equirectangular 360 LoRA v3",
        "repo_id": "MultiTrickFox/Flux-LoRA-Equirectangular-v3",
        "hf_path": "equirectangular_flux_lora_v3_000003072.safetensors",
        "subdir": "loras",
        "filename": "equirectangular_flux_lora_v3_000003072.safetensors",
        "local_dir": _LORA_LOCAL_DIR,  # bypass GCS symlink
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


def _register_extra_paths():
    """Register /tmp model dirs so ComfyUI loaders can find them."""
    try:
        import folder_paths
        folder_paths.add_model_folder_path("vae", _VAE_LOCAL_DIR)
        folder_paths.add_model_folder_path("loras", _LORA_LOCAL_DIR)
        folder_paths.add_model_folder_path("upscale_models", _UPSCALE_LOCAL_DIR)
    except Exception as e:
        _log(f"WARNING: could not register extra paths with folder_paths: {e}")


_download_models()
_register_extra_paths()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
