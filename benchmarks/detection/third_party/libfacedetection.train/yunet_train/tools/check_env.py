from __future__ import annotations

import importlib

import torch


def main() -> None:
    print(f"torch: {torch.__version__}")
    print(f"cuda: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")

    for name in ("numpy", "cv2", "tqdm", "yaml"):
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"{name}: {version}")


if __name__ == "__main__":
    main()

