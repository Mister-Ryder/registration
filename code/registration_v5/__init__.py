"""Registration V5: frozen DINO anchor plus capture-aware dense correction.

V5 consolidates the successful V4-Core descriptor, the audited dense-corefix
solver, and the label-free capture router into one publication-facing package.
No implementation is imported from an experimental ``registration_v4_*``
directory.  The official DINO-Reg repository remains a pinned external model
dependency, just like its foundation-model weights.
"""

__version__ = "5.0.0-dino-anchor-capture-corefix-20260827"

__all__ = [
    "CaptureDecision",
    "CaptureRouterConfig",
    "CorefixResult",
    "SolverCorefixConfig",
    "V5Protocol",
    "decide_from_corefix_qa",
    "load_protocol",
    "solve_corefix",
]


def __getattr__(name):
    """Keep lightweight provenance/routing imports usable without GPU extras."""
    if name in {"V5Protocol", "load_protocol"}:
        from .protocol import V4FinalProtocol, load_protocol

        return V4FinalProtocol if name == "V5Protocol" else load_protocol
    if name in {"CaptureDecision", "CaptureRouterConfig", "decide_from_corefix_qa"}:
        from .routing import capture

        return getattr(capture, name)
    if name in {"CorefixResult", "SolverCorefixConfig", "solve_corefix"}:
        from .solver import corefix

        return getattr(corefix, name)
    raise AttributeError(name)
