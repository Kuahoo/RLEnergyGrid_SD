from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sdge_microgrid.constants import EPISODE_HOURS

RHO_FILES = {"normal": "rho_normal", "highrisk": "rho_highrisk"}


@dataclass(frozen=True)
class WeekTrace:
    pv: np.ndarray
    pcrit: np.ndarray
    pflex: np.ndarray
    rho: np.ndarray
    start_hour: int

    def __post_init__(self) -> None:
        for name in ("pv", "pcrit", "pflex", "rho"):
            if getattr(self, name).shape != (EPISODE_HOURS,):
                raise ValueError(f"{name} must have shape ({EPISODE_HOURS},)")


def load_week(
    directory: Path | str,
    start_hour: int,
    scenario: str = "normal",
) -> WeekTrace:
    if scenario not in RHO_FILES:
        raise ValueError(f"Unknown scenario {scenario!r}; expected {sorted(RHO_FILES)}")
    d = Path(directory)
    sl = slice(start_hour, start_hour + EPISODE_HOURS)
    return WeekTrace(
        pv=np.load(d / "pv.npy")[sl],
        pcrit=np.load(d / "pcrit.npy")[sl],
        pflex=np.load(d / "pflex.npy")[sl],
        rho=np.load(d / f"{RHO_FILES[scenario]}.npy")[sl],
        start_hour=int(start_hour),
    )
