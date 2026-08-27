"""Legacy B05 descriptor probe using the frozen 24-channel ConvexAdam cost."""

from __future__ import annotations

from . import probe_legacy_b05_convex as _entry
from ..convex_solver_v2 import descriptor_convex_adam_v2


def main(argv=None) -> int:
    _entry.descriptor_convex_adam = descriptor_convex_adam_v2
    return _entry.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
