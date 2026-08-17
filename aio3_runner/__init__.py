"""AIO3-v1 data, model-adapter, metric and evaluation support for DACG-IR.

Submodules are intentionally not imported eagerly. This keeps manifest auditing
usable on data-preparation machines that do not have the full model stack.
"""

__all__ = ["data", "metrics", "adapter", "results"]
