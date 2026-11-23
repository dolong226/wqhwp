"""Shared paths and output helper for the scripts."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
CONFIGS, RESULTS = ROOT / "configs", ROOT / "results"
sys.path.insert(0, str(ROOT))


def load_config(name):
    return json.loads((CONFIGS / name).read_text())


def write_result(name, script, config_name, payload):
    """Write results/<name> with provenance: SHA-256 of the script and config, library versions."""
    meta = {"script": Path(script).name,
            "script_sha256": hashlib.sha256(Path(script).read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256((CONFIGS / config_name).read_bytes()).hexdigest() if config_name else None,
            "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__}}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps({**meta, **payload}, indent=1))
    print("wrote", RESULTS / name)
