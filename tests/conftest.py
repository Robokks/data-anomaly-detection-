import os

# Must be set before any Qt module is imported anywhere in the test session,
# so GUI tests (and anything importing gui.*) run headlessly in this
# container -- no real display is available.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
