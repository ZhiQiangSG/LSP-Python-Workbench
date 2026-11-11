from __future__ import annotations
import json, hashlib, platform
from pathlib import Path
from datetime import datetime, timezone

def hash_object(obj):
    raw = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()

class AuditLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
    def write(self, event, **details):
        row = {"timestamp": datetime.now(timezone.utc).isoformat(),
               "event": event, **details}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")

def new_manifest(project_name, base_model, base_revision=None, tokenizer=None,
                 harness_revision=None, seed=None):
    return {
        "schema_version": "1.0",
        "project_name": project_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": {"id": base_model, "revision": base_revision, "sha256": None},
        "tokenizer": tokenizer,
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "harness_revision": harness_revision,
        "seed": seed,
        "method": "LSP",
        "programmer": {},
        "candidate_pool": [],
        "selection": {},
        "budgets": {},
        "safety": {},
        "retention": {},
        "rollback": {},
        "artifacts": {}
    }

def write_manifest(manifest, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
