# e_training/utils_device.py
import torch

def pick_device(pref: str | None = "auto"):
    pref = (pref or "auto").lower()
    if pref == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps"), "mps"
        return torch.device("cpu"), "cpu"
    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        return torch.device("cpu"), "cpu"
    # auto
    if torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    return torch.device("cpu"), "cpu"