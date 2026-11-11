from __future__ import annotations
import hashlib, json
from .schemas import Candidate
from .directions import footprint

ALLOWED_SUFFIXES = {"o_proj", "down_proj"}

def eligible_block_indices(n_blocks: int, fraction=(0.35, 0.65)):
    lo = max(0, int(n_blocks * fraction[0]))
    hi = min(n_blocks, int(n_blocks * fraction[1]))
    return list(range(lo, max(lo + 1, hi)))

def module_is_allowed(module_name: str, target_suffixes=ALLOWED_SUFFIXES):
    return module_name.split(".")[-1] in target_suffixes

def make_candidate_pool(category, module_specs, singular_values_by_module,
                        n_blocks, block_fraction=(0.35, 0.65),
                        max_candidates=12, target_suffixes=ALLOWED_SUFFIXES):
    allowed_blocks = set(eligible_block_indices(n_blocks, block_fraction))
    rows = []
    for spec in module_specs:
        name = spec["name"]
        if not module_is_allowed(name, target_suffixes):
            continue
        if int(spec["block_index"]) not in allowed_blocks:
            continue
        score = float(spec.get("diagnostic_score", 0))
        sv = [float(x) for x in singular_values_by_module.get(name, [])]
        for sign in (-1, 1):
            cid = hashlib.sha1(f"{category}|{name}|{sign}".encode()).hexdigest()[:10]
            rows.append(Candidate(
                candidate_id=cid, category=category, module_name=name,
                block_index=int(spec["block_index"]), sign=sign,
                diagnostic_score=score, d_in=int(spec["d_in"]),
                d_out=int(spec["d_out"]),
                footprint=footprint(1, spec["d_in"], spec["d_out"]),
                singular_values=sv
            ))
    rows.sort(key=lambda c: (-c.diagnostic_score, c.module_name, c.sign))
    return rows[:max_candidates]

def enforce_footprint(candidates, base_parameter_count, fraction=0.001):
    cap = int(base_parameter_count * fraction)
    selected, used = [], 0
    for c in candidates:
        if used + c.footprint <= cap:
            selected.append(c)
            used += c.footprint
    return selected, {"used": used, "cap": cap, "fraction": fraction}
