import os

# Must be set before any Qt module is imported anywhere in the test session,
# so GUI tests (and anything importing gui.*) run headlessly in this
# container -- no real display is available.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Must be set before torch is imported anywhere in the test session. A
# GPU-enabled torch wheel (the default from PyPI) probing CUDA from a
# background QThread reliably SIGABRTs in this container -- and more
# broadly, this app is CPU-only by design (AutoencoderDetector defaults to
# device="cpu", no GPU assumed), so torch should never touch CUDA here
# regardless of which wheel happens to be installed.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
