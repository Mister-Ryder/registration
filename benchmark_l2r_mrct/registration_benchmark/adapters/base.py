"""Adapter interface and fail-closed method dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Protocol, Tuple

import numpy as np

from ..contract import PairTask


@dataclass
class AdapterResult:
    flow_dzyx: np.ndarray
    diagnostics: Dict[str, Any]


class MethodAdapter(Protocol):
    def register(self, task: PairTask, config: Mapping[str, Any], work_dir: Path) -> AdapterResult:
        ...


def create_adapter(name: str) -> MethodAdapter:
    key = name.strip().lower().replace("-", "_")
    if key == "identity":
        from .identity import IdentityAdapter
        return IdentityAdapter()
    if key in {"ants", "ants_mi", "ants_syn_mi"}:
        from .ants import ANTsAdapter
        return ANTsAdapter()
    if key in {"convexadam", "convexadam_mind"}:
        from .convexadam import ConvexAdamAdapter
        return ConvexAdamAdapter()
    if key in {"fireants", "fireants_mi"}:
        from .fireants import FireANTsAdapter
        return FireANTsAdapter()
    if key in {"locor"}:
        from .locor import LocorAdapter
        return LocorAdapter()
    if key in {"dino_reg", "dinoreg"}:
        from .dino_reg import DINORegAdapter
        return DINORegAdapter()
    if key in {"mok_dns_io", "dns_io"}:
        from ..dns.io_registration import DNSIOAdapter
        return DNSIOAdapter()
    if key in {"ours", "registration_v3_pracm"}:
        from .ours import OursAdapter
        return OursAdapter()
    if key in {"external", "synmse", "dgmir_u", "m2m_reg", "transmorph_mind", "corrmlp_mind"}:
        from .external import ExternalCommandAdapter
        return ExternalCommandAdapter(method_name=key)
    raise ValueError(f"Unknown method adapter: {name!r}")
