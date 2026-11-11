from __future__ import annotations
import json, hashlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable
import numpy as np
from .schemas import Trajectory

CATEGORIES = ("malformed_tool_call", "looping", "token_exhaustion")

def _has_progress(events, start, end):
    # Events are ordered; span indices are token offsets. When event offsets are
    # unavailable, callers should provide explicit spans from the harness.
    for ev in events:
        if ev.state_changed or ev.changed_next_action or ev.successful_termination:
            return True
    return False

def detect_repeated_8grams(token_ids: list[int], window: int = 256,
                           min_repeats: int = 3, ngram: int = 8):
    """Return half-open token spans containing a repeated 8-gram >= min_repeats.
    This is a token-level detector; progress exclusion is applied by trajectory
    ingestion where event-to-token alignment is available.
    """
    spans = []
    for left in range(0, max(1, len(token_ids) - window + 1)):
        right = min(len(token_ids), left + window)
        counts = defaultdict(list)
        for i in range(left, max(left, right - ngram + 1)):
            counts[tuple(token_ids[i:i+ngram])].append(i)
        repeated = [pos for pos in counts.values() if len(pos) >= min_repeats]
        if repeated:
            lo = min(p[0] for p in repeated)
            hi = max(p[-1] + ngram for p in repeated)
            span = (lo, hi)
            if not spans or span[0] > spans[-1][1]:
                spans.append(span)
            else:
                spans[-1] = (spans[-1][0], max(spans[-1][1], span[1]))
    return spans

def label_trajectory(t: Trajectory) -> list[str]:
    labels = set(t.failures)
    if t.output_limit_hit:
        labels.add("token_exhaustion")
    if any(e.type == "tool_call" and e.valid is False for e in t.events):
        labels.add("malformed_tool_call")
    if "looping" not in t.spans:
        spans = detect_repeated_8grams(t.token_ids)
        # Conservative: require no progress evidence in the trajectory.
        if spans and not any(_has_progress(t.events, *s) for s in spans):
            labels.add("looping")
    elif t.spans["looping"]:
        labels.add("looping")
    return sorted(labels)

def load_trajectories(path: str | Path) -> list[Trajectory]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                t = Trajectory.model_validate_json(line)
                t.failures = label_trajectory(t)
                rows.append(t)
    return rows

def save_trajectories(rows: Iterable[Trajectory], path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")

def repository_grouped_folds(rows: list[Trajectory], n_splits=5, seed=0):
    """Deterministic repository-grouped outer folds; no repository leakage."""
    repos = sorted({r.repository for r in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(repos)
    folds = [[] for _ in range(min(n_splits, len(repos)))]
    for i, repo in enumerate(repos):
        folds[i % len(folds)].append(repo)
    result = []
    for heldout in folds:
        test_repos = set(heldout)
        test = [r for r in rows if r.repository in test_repos]
        train = [r for r in rows if r.repository not in test_repos]
        result.append({"train": train, "test": test,
                       "heldout_repositories": sorted(test_repos)})
    return result

def nested_repository_split(rows: list[Trajectory], seed=0):
    """Split outer-train repositories into diagnostic and validation groups."""
    repos = sorted({r.repository for r in rows})
    if len(repos) < 2:
        raise ValueError("At least two repositories are required for grouped diagnostic/validation split.")
    rng = np.random.default_rng(seed)
    rng.shuffle(repos)
    cut = max(1, len(repos) // 2)
    diag_repos, val_repos = set(repos[:cut]), set(repos[cut:])
    if not val_repos:
        val_repos.add(diag_repos.pop())
    return ([r for r in rows if r.repository in diag_repos],
            [r for r in rows if r.repository in val_repos])

def task_equal_activation_differences(records: dict, category: str):
    """Aggregate matched failing-successful activation means, equally per task.
    records keys: task_id, category, role, progress_bin, activations.
    activations are [rows, d_out] arrays or [rows, tokens, d_out].
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for i, (task, cat, role, progress, act) in enumerate(zip(
        records["task_id"], records["category"], records["role"],
        records["progress_bin"], records["activations"]
    )):
        if str(cat) != category:
            continue
        a = np.asarray(act, dtype=np.float32)
        if a.ndim == 1:
            mean = a
        else:
            mean = a.reshape(-1, a.shape[-1]).mean(axis=0)
        grouped[str(task)][(str(role), int(progress))].append(mean)
    task_diffs = []
    for task, buckets in grouped.items():
        common = sorted({p for role, p in buckets if role == "failing"} &
                        {p for role, p in buckets if role == "successful"})
        if not common:
            continue
        diffs = []
        for p in common:
            f = np.stack(buckets[("failing", p)]).mean(axis=0)
            s = np.stack(buckets[("successful", p)]).mean(axis=0)
            diffs.append(f - s)
        task_diffs.append(np.stack(diffs).mean(axis=0))
    return np.stack(task_diffs) if task_diffs else np.empty((0, 0), np.float32)

def load_activation_npz(path: str | Path):
    z = np.load(path, allow_pickle=False)
    required = {"task_id", "category", "role", "progress_bin", "activations"}
    missing = required - set(z.files)
    if missing:
        raise ValueError(f"Activation file missing keys: {sorted(missing)}")
    return {k: z[k] for k in required | ({"token_mask"} if "token_mask" in z.files else set())}

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
