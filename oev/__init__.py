"""OEV: a small System One decision model.

State + typed questions in, calibrated probability distributions out,
one forward pass.
"""

from oev.infer import OEV

__version__ = "0.3.0"
__all__ = ["OEV", "__version__"]
