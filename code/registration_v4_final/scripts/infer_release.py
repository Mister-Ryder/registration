"""Formal inference entry using the independently audited DNS-Convex solver."""

from __future__ import annotations

from . import infer as _entry
from ..convex_solver_release import descriptor_convex_adam_release
from ..network_release import V4FinalRegistrationModelRelease


def main(argv=None) -> int:
    _entry.V4FinalModel = V4FinalRegistrationModelRelease
    _entry.descriptor_convex_adam = descriptor_convex_adam_release
    return _entry.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
