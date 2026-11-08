"""Write experiment_manifest.json: code hash, versions, conventions, split IDs, run configs.

    python scripts/write_manifest.py --runs <RUNS dir> --out <RUNS dir>/experiment_manifest.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from camera_motion import dataset_generator as D  # noqa: E402
from camera_motion.config import FROZEN, METRICS_VERSION  # noqa: E402


def code_hash():
    h = hashlib.sha256()
    files = sorted(glob.glob(os.path.join(ROOT, "camera_motion", "*.py")) + glob.glob(os.path.join(ROOT, "scripts", "*.py")))
    for f in files:
        h.update(os.path.relpath(f, ROOT).encode())
        h.update(open(f, "rb").read())
    return h.hexdigest()[:16], [os.path.relpath(f, ROOT) for f in files]


def split_fingerprint(ds):
    ids = np.unique(ds["condition_id"])
    pub = D.public_conditions(ds)
    h = hashlib.sha256(np.ascontiguousarray(pub["start_pose"]).tobytes() + np.ascontiguousarray(pub["subject"]).tobytes())
    return {"n_conditions": int(len(ids)), "first_id": str(ids[0]), "last_id": str(ids[-1]), "public_hash": h.hexdigest()[:16]}


def main(runs, out):
    ch, files = code_hash()
    try:
        import subprocess
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None
    except Exception:
        git = None
    man = {
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "code_sha256_16": ch, "git_revision": git, "files_hashed": files,
        "metrics_version": METRICS_VERSION, "classifier_version": FROZEN.clf.version,
        "config": FROZEN.to_dict(), "python": platform.python_version(),
        "splits": {
            f"{fam}_{panel}": split_fingerprint(D.generate_e1_eval(fam, panel, n_ref=1))
            for fam, panel in (("E1a", "eval_id"), ("E1a", "eval_ood_position"), ("E1b", "eval_id"))
        },
        "runs": {},
    }
    for meta_f in sorted(glob.glob(os.path.join(runs, "*", "run_meta.json"))):
        d = os.path.dirname(meta_f)
        entry = {"run_meta": json.load(open(meta_f))}
        if os.path.exists(os.path.join(d, "train_summary.json")):
            entry["train_summary"] = json.load(open(os.path.join(d, "train_summary.json")))
        entry["evals"] = sorted(os.path.basename(e) for e in glob.glob(os.path.join(d, "eval_*")))
        man["runs"][os.path.basename(d)] = entry
    with open(out, "w") as f:
        json.dump(man, f, indent=2)
    print("wrote", out, "| code", ch, "| runs:", list(man["runs"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    x = ap.parse_args()
    main(x.runs, x.out)
