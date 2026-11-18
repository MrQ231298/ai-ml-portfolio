"""Download, verify and load NASA C-MAPSS (turbofan run-to-failure simulation).

Source: NASA Prognostics Center of Excellence data repository. US Government work,
public domain. Citation: A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage
Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM08, 2008.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / "downloads"
URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
SHA256 = "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"
COLS = ["unit", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]
SEED = 21


def fetch() -> Path:
    """Download the outer zip once, check its SHA-256, extract the inner CMAPSSData.zip."""
    out = DOWNLOADS / "CMAPSSData"
    if (out / "train_FD001.txt").exists():
        return out
    DOWNLOADS.mkdir(exist_ok=True)
    z = DOWNLOADS / "cmapss.zip"
    if not z.exists():
        urllib.request.urlretrieve(URL, z)
    digest = hashlib.sha256(z.read_bytes()).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"checksum mismatch: {digest}")
    with zipfile.ZipFile(z) as outer:
        inner = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
        with zipfile.ZipFile(io.BytesIO(outer.read(inner))) as zz:
            zz.extractall(out)
    return out


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
    df[["unit", "cycle"]] = df[["unit", "cycle"]].astype(int)
    return df


def load(fd: str = "FD001") -> dict:
    """train: full run-to-failure trajectories with RUL; test: truncated trajectories; test_rul: truth per test unit."""
    d = fetch()
    train = _read(d / f"train_{fd}.txt")
    train["rul"] = train.groupby("unit").cycle.transform("max") - train.cycle
    test = _read(d / f"test_{fd}.txt")
    rul = pd.read_csv(d / f"RUL_{fd}.txt", header=None, names=["rul"]).rul.to_numpy()
    return {"train": train, "test": test, "test_rul": rul}


def split_units(units, seed: int = SEED, frac=(0.6, 0.2, 0.2)) -> dict:
    """Split ENGINES (never cycles) into fit / validation / held-out test, so no trajectory is in two splits."""
    u = np.array(sorted(units))
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    a, b = int(frac[0] * len(u)), int((frac[0] + frac[1]) * len(u))
    return {"fit": sorted(u[:a].tolist()), "val": sorted(u[a:b].tolist()), "test": sorted(u[b:].tolist())}
