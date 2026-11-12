"""Parity: statistical conformance testing for LLM inference endpoints."""

__version__ = "0.3.0"

from parity.suite import Reference, Suite, Target, hf_reference

__all__ = ["Reference", "Suite", "Target", "__version__", "hf_reference"]
