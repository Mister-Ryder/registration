"""Authoritative V4-final inference entry."""

from __future__ import annotations

from . import infer as _legacy_entry
from ..convex_solver_v2 import descriptor_convex_adam_v2
from ..network import V4FinalRegistrationModel


def main(argv=None) -> int:
    _legacy_entry.V4FinalModel = V4FinalRegistrationModel
    _legacy_entry.descriptor_convex_adam = descriptor_convex_adam_v2
    return _legacy_entry.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

