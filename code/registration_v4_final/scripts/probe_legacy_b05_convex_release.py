"""Legacy B05 descriptor probe using the independently audited solver."""

from __future__ import annotations

from . import probe_legacy_b05_convex as _entry
from ..convex_solver_release import descriptor_convex_adam_release


def main(argv=None) -> int:
    _entry.descriptor_convex_adam = descriptor_convex_adam_release
    return _entry.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
