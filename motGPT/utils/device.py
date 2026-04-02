import warnings
import torch


def resolve_torch_device(accelerator: str) -> torch.device:
    """
    Map config ACCELERATOR ('gpu' / 'cpu' / etc.) to a torch.device.
    If 'gpu' is set but CUDA is unavailable (e.g. CPU-only PyTorch), use CPU and warn once.
    """
    if accelerator == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        warnings.warn(
            "ACCELERATOR is 'gpu' but CUDA is not available; using CPU. "
            "Install a CUDA build of PyTorch from https://pytorch.org/ "
            "or set ACCELERATOR to 'cpu' in your config (e.g. configs/webui.yaml).",
            UserWarning,
            stacklevel=2,
        )
        return torch.device("cpu")
    return torch.device("cpu")
