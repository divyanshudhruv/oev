"""OEV: a small System One decision model."""

__version__ = "0.3.0"
__all__ = ["OEV", "__version__"]


def __getattr__(name):
    # OEV needs torch; keep it lazy so torch-free consumers (onnxruntime
    # serving, oev.presets, packers) can import oev.* without torch.
    if name == "OEV":
        from oev.infer import OEV

        return OEV
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
