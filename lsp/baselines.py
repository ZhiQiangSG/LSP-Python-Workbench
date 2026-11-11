from __future__ import annotations
from .evaluation import summarize_results

def fixed_directional_baseline(candidates, category_prevalence):
    """C1: highest-prevalence category, highest-score candidate, rank1/dose1."""
    if not candidates: return []
    category = sorted(category_prevalence, key=lambda k:(-category_prevalence[k], k))[0]
    eligible = [c for c in candidates if c.category == category]
    if not eligible: return []
    chosen = sorted(eligible, key=lambda c:(-c.diagnostic_score,c.module_name,c.sign))[0]
    return [{"candidate_id":chosen.candidate_id,"rank":1,"dose":1.0}]

def deterministic_greedy(candidates, evaluate, baseline_results, budget,
                          max_items=None):
    """C2: deterministic forward selection over identical candidate pool.
    evaluate(selection) must return task-level rows and consume/account rollouts.
    """
    selected, remaining = [], list(candidates)
    current = baseline_results
    while remaining and budget.remaining > 0 and (max_items is None or len(selected)<max_items):
        scored = []
        for c in remaining:
            trial = selected + [{"candidate_id":c.candidate_id,"rank":1,"dose":1.0}]
            rows = evaluate(trial)
            score = summarize_results(current)["pathological_tasks"] - summarize_results(rows)["pathological_tasks"]
            scored.append((score,c.diagnostic_score,c.module_name,c.sign,c,rows))
        scored.sort(key=lambda x:(-x[0],-x[1],x[2],x[3]))
        best = scored[0]
        if best[0] <= 0: break
        selected.append({"candidate_id":best[4].candidate_id,"rank":1,"dose":1.0})
        current = best[5]
        remaining.remove(best[4])
    return {"selection":selected,"results":current,"rollouts_used":budget.used}
