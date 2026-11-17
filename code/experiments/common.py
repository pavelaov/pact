"""Shared harness for every PACT experiment.

Conventions that make the paper reproducible:
  * every experiment takes an explicit integer master seed (default in SEEDS);
  * every experiment writes  out/<name>.csv  and  out/<name>.json  (metadata:
    seed, parameters, wall-clock, environment) and appends to logs/<name>.log;
  * no experiment ever reads a value produced by another experiment except
    through those files.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # pact/
CODE = os.path.dirname(HERE)                            # pact/code
OUT = os.path.join(ROOT, "out")
LOGS = os.path.join(ROOT, "logs")

sys.path.insert(0, CODE)

os.makedirs(OUT, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

# One master seed per experiment.  Changing these changes every number in the
# paper, so they are pinned here and echoed into every output file.
SEEDS = {
    "exp1_correctness": 20240001,
    "exp2_detection": 20240002,
    "exp3_yield": 20240003,
    "exp4_adaptive": 20240004,
    "exp5_defamation": 20240005,
    "exp6_consistency": 20240006,
    "exp7_frontier": 20240007,
    "exp8_auditcost": 20240008,
    "exp9_backdoor": 20240009,
    "exp10_costmicro": 20240010,
    "exp11_distinguisher": 20240011,
    "exp12_singlechip": 20240012,
}


def _relpath(path: str) -> str:
    """Log paths relative to the artifact root, with forward slashes.

    Logs ship with the artifact, so an absolute path would carry the author's
    directory layout (and often a user name) into a double-blind submission.
    """
    try:
        return os.path.relpath(path, ROOT).replace(os.sep, "/")
    except ValueError:                       # different drive on Windows
        return os.path.basename(path)


def env_metadata() -> Dict[str, Any]:
    try:
        import cryptography
        cver = cryptography.__version__
    except Exception:
        cver = "unavailable"
    try:
        import numpy, scipy, matplotlib
        nver, sver, mver = numpy.__version__, scipy.__version__, matplotlib.__version__
    except Exception:
        nver = sver = mver = "unavailable"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cryptography": cver,
        "numpy": nver,
        "scipy": sver,
        "matplotlib": mver,
    }


def _default(o):
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, (bytes, bytearray)):
        return o.hex()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def write_csv(name: str, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str] = None) -> str:
    path = os.path.join(OUT, f"{name}.csv")
    if not rows:
        open(path, "w").close()
        return path
    fieldnames = list(fieldnames or rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    return path


def write_json(name: str, obj: Any) -> str:
    path = os.path.join(OUT, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_default)
    return path


class Run:
    """Context manager that logs, times and stamps one experiment."""

    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = dict(params or {})
        self.seed = SEEDS.get(name, 1)
        self.params.setdefault("seed", self.seed)
        self.t0 = 0.0
        self.log_path = os.path.join(LOGS, f"{name}.log")
        self._fh = None

    def __enter__(self) -> "Run":
        self.t0 = time.time()
        self._fh = open(self.log_path, "w", encoding="utf-8")
        self.log(f"=== {self.name} ===")
        self.log(f"started {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"seed    {self.seed}")
        self.log(f"params  {json.dumps(self.params, default=_default)}")
        self.log(f"env     {json.dumps(env_metadata())}")
        return self

    def log(self, msg: str) -> None:
        line = str(msg)
        print(line, flush=True)
        if self._fh:
            self._fh.write(line + "\n")
            self._fh.flush()

    def finish(self, rows: Sequence[Dict[str, Any]] = (), summary: Dict[str, Any] = None,
               fieldnames: Sequence[str] = None) -> None:
        elapsed = time.time() - self.t0
        meta = {
            "experiment": self.name,
            "seed": self.seed,
            "params": self.params,
            "env": env_metadata(),
            "wall_clock_s": round(elapsed, 3),
            "n_rows": len(rows),
            "summary": summary or {},
        }
        if rows:
            p = write_csv(self.name, rows, fieldnames)
            self.log(f"wrote {_relpath(p)} ({len(rows)} rows)")
        p = write_json(self.name, meta)
        self.log(f"wrote {_relpath(p)}")
        self.log(f"elapsed {elapsed:.2f} s")

    def __exit__(self, *exc):
        if self._fh:
            self._fh.close()
        return False
