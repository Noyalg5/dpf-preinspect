"""Vision pre-inspection and downstream-outcome prediction for diesel particulate filters."""

import os

# Some torchvision ops are not yet implemented on Apple MPS; fall back to CPU for those.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.1.0"
