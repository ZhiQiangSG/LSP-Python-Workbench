from __future__ import annotations
import json, os, typer, yaml
from pathlib import Path
from rich import print
from rich.table import Table
from .manifest import new_manifest, write_manifest, AuditLog
from .data import load_trajectories, repository_grouped_folds
from .directions import projection_update, factorize_update, lora_dense_delta
from .schemas import Candidate
from .programmer import OpenAICompatibleProgrammer, ConstrainedWorkbench, TOOL_SCHEMAS

app = typer.Typer(no_args_is_help=True, help="LSP Python Workbench")

@app.command()
def demo(out: str = typer.Option("runs/demo", help="Output directory")):
    """Run an offline synthetic workflow demonstrating factors, gating and manifest."""
    import numpy as np
    outp = Path(out); outp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    w = rng.normal(size=(16, 12)).astype("float32")
    p, _ = np.linalg.qr(rng.normal(size=(16, 1)))
    delta = projection_update(w, p.astype("float32"), 1.0)
    a, b, err = factorize_update(delta, rank=1, alpha=1, dose=1)
    reconstructed = lora_dense_delta(a,b,1,1,1)
    manifest = new_manifest("LSP Python Workbench", "synthetic/student-demo", seed=7)
    manifest["selection"] = {"demo":True,"rank":1,"dose":1.0}
    manifest["rollback"] = {"factorization_error":err,
                            "dense_delta_max_abs_error":float(np.max(np.abs(delta-reconstructed))),
                            "base_modified":False}
    write_manifest(manifest, outp/"manifest.json")
    AuditLog(outp/"audit.jsonl").write("demo_completed", factorization_error=err)
    print(f"[green]Offline demo completed.[/green] Manifest: {outp/'manifest.json'}")
    print(f"Factorization relative error: {err:.3e}")

@app.command()
def inspect(model_id: str = typer.Option("google/gemma-4-e2b-it"),
            revision: str = typer.Option(None),
            dtype: str = typer.Option("bfloat16")):
    """Load student checkpoint and list eligible linear projections."""
    from .student import load_student, discover_linear_modules
    model, tokenizer = load_student(model_id, revision=revision, dtype=dtype)
    rows = discover_linear_modules(model)
    table = Table("Module","Block","d_in","d_out")
    for r in rows:
        if r["name"].split(".")[-1] in {"o_proj","down_proj"}:
            table.add_row(r["name"],str(r["block_index"]),str(r["d_in"]),str(r["d_out"]))
    print(table)
    print(f"Tokenizer: {tokenizer.__class__.__name__}; linear modules: {len(rows)}")

@app.command()
def folds(trajectory_file: str, n_splits: int = 5, seed: int = 20261112):
    """Print repository-grouped outer fold composition."""
    rows = load_trajectories(trajectory_file)
    result = repository_grouped_folds(rows,n_splits,seed)
    for i, fold in enumerate(result):
        print({"fold":i,"train_tasks":len(fold["train"]),
               "test_tasks":len(fold["test"]),
               "heldout_repositories":fold["heldout_repositories"]})

@app.command()
def program(config: str = typer.Option("configs/experiment.yaml"),
            out: str = typer.Option("runs/frontier")):
    """Run the frontier programmer against a precomputed fixed candidate pool.
    Candidate JSON and validation summaries are read from out/inputs/.
    """
    outp = Path(out); inp = outp/"inputs"
    cand_data = json.loads((inp/"candidates.json").read_text())
    val = json.loads((inp/"validation_summaries.json").read_text())
    candidates = [Candidate.model_validate(x) for x in cand_data]
    cfg = yaml.safe_load(Path(config).read_text())
    cap = int(cfg["student"].get("base_parameter_count", 0) *
              cfg["experiment"]["adapter_footprint_fraction"]) or None
    workbench = ConstrainedWorkbench(
        candidates, val,
        allowed_ranks=cfg["experiment"]["ranks"],
        allowed_doses=cfg["experiment"]["dose_grid"],
        footprint_cap=cap,
        audit_log=outp/"audit.jsonl")
    programmer = OpenAICompatibleProgrammer(
        base_url=os.getenv("LSP_BASE_URL", cfg["programmer"].get("base_url")),
        model=os.getenv("LSP_MODEL", cfg["programmer"].get("model")),
        temperature=cfg["programmer"].get("temperature",0),
        max_tool_rounds=cfg["programmer"].get("max_tool_rounds",12))
    brief = {"project":cfg.get("project_name"),"constraints":cfg["experiment"],
             "candidates":[c.model_dump() for c in candidates],
             "validation_summaries":val,
             "instruction":"Select only candidates justified by validation evidence. If acceptance criteria are unmet, return an empty candidate list."}
    result = programmer.decide(brief, TOOL_SCHEMAS, workbench.dispatch)
    selection = result.get("selection") or workbench.selection or {"candidate_ids":[]}
    (outp).mkdir(parents=True,exist_ok=True)
    (outp/"programmer_result.json").write_text(json.dumps(
        {"selection":selection,"transcript":result.get("transcript",[])},
        indent=2,default=str))
    print(selection)

if __name__ == "__main__":
    app()
