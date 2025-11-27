# e_training/utils_device.py
import torch

def pick_device(pref: str | None = "auto"):
    """
    Selects the best available torch device.
    Priority: CUDA > MPS > CPU unless explicitly requested.
    Returns (torch.device, str_name).
    """
    pref = str(pref or "auto").strip().lower()

    # Explicit user preferences
    if pref == "cuda":
        return (torch.device("cuda"), "cuda") if torch.cuda.is_available() else (torch.device("cpu"), "cpu")
    if pref == "mps":
        return (torch.device("mps"), "mps") if torch.backends.mps.is_available() else (torch.device("cpu"), "cpu")
    if pref == "cpu":
        return torch.device("cpu"), "cpu"

    # Auto-detect: CUDA > MPS > CPU
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"