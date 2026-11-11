# LSP Python Workbench

A sandboxed, auditable Python workbench for frontier-model-driven LoRA-space programming (LSP) experiments described in *Programming the Adapter, Not the Base*.

The workbench treats the frontier model as a **programmer and analyst**, not as the student model. It exposes a constrained tool API for inspecting Gemma 4 e2B, extracting activation evidence, constructing candidate low-rank updates, evaluating candidate adapters, and producing manifests. The base model is never modified in place.

> Research status: this is an experimental scaffold, not a validated implementation of the paper's full evaluation. Validate the installed Gemma 4 architecture, hook points, tokenizer, harness integration, and numerical tolerances before confirmatory runs.

## Design goals

- Keep the student model frozen and make all interventions removable.
- Separate diagnostic, validation, and held-out evaluation data.
- Let any frontier model operate through a provider-neutral JSON tool protocol.
- Restrict the programmer to a fixed candidate pool and explicit rollout/parameter budgets.
- Audit every inspection and mutation; prohibit optimizer/autograd-based training.
- Keep the deterministic greedy baseline and fixed-rule baseline implementable with the same candidate pool.
- Record provenance, budgets, decisions, safety/retention gates, and rollback checks in a machine-readable manifest.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                   # optional; set provider credentials here
```

Run the offline demo (no model download, API key, or GPU required):

```bash
python -m lsp demo --out runs/demo
```

Run tests:

```bash
pytest
```

Inspect the CLI:

```bash
python -m lsp --help
```

## Architecture

```text
Frontier model (any provider / local model)
              |
       provider adapter
              |
     constrained tool loop
              |
     Workbench API / policy
       |       |        |
   evidence  candidate  evaluation
   store     registry   runner
       |       |        |
       +---- manifest / audit log ----+
                                      |
                              removable LoRA
                                      |
                          frozen Gemma 4 e2B
```

### Components

- `lsp.student`: Hugging Face model loading, architecture discovery, activation capture, adapter attach/detach.
- `lsp.data`: trajectory labels, matched spans, task-grouped folds, activation-difference aggregation.
- `lsp.directions`: thin SVD, projection updates, LoRA factorization and reconstruction checks.
- `lsp.candidates`: middle-30%-block restriction, residual-writing module filter, candidate pool and footprint cap.
- `lsp.evaluation`: task rollout budgets, pathology metrics, paired comparisons, retention/safety gates.
- `lsp.programmer`: provider-neutral tool protocol and constrained agent loop.
- `lsp.baselines`: fixed directional and deterministic greedy baselines.
- `lsp.manifest`: reproducibility manifest and append-only JSONL audit events.

## Frontier model integration

The workbench does not require a particular frontier model or vendor. Implement `BaseProgrammer` for the provider of choice, or use the OpenAI-compatible adapter if the provider exposes a compatible chat-completions endpoint.

The model receives:
1. a compact experiment brief;
2. candidate summaries and validation results;
3. only the tool functions listed in `lsp.programmer.TOOL_SCHEMAS`.

It can call tools to inspect candidates, compare validation results, select candidates, set rank/dose, and finalize. It cannot supply arbitrary Python, arbitrary file paths, or expand the candidate pool. Python execution is performed by the local workbench, not generated and executed from model text.

Example:

```bash
export LSP_PROVIDER=openai_compatible
export LSP_BASE_URL=https://your-provider.example/v1
export LSP_API_KEY=...
export LSP_MODEL=your-frontier-model
python -m lsp program --config configs/experiment.yaml --out runs/frontier
```

Provider configuration can also be supplied in YAML. Never put API keys in the YAML or commit `.env`.

## Experiment lifecycle

1. **Inspect** the pinned student model and record its architecture, module names, hashes, tokenizer and runtime.
2. **Ingest** diagnostic trajectories and activation captures. The detector uses the paper's definitions: repeated token 8-grams at least three times in a rolling 256-token window without state progress; malformed calls; output-limit exhaustion.
3. **Construct** per-task equally weighted failing-minus-matched-successful activation differences and category-specific thin-SVD subspaces.
4. **Generate** up to 12 ranked candidates per supported category, limited to the middle 30% of blocks and `o_proj` / `down_proj`.
5. **Screen** candidates at rank 1, dose 1.0 on eight validation tasks (maximum 96 candidate rollouts when there are 12 candidates).
6. **Select** within the per-method 300-rollout validation budget. The default acceptance gate requires at least two fewer pathological validation tasks, no more than one fewer resolved task, and passing retention/safety thresholds.
7. **Export** a standalone adapter plus a manifest. Keep the base checkpoint unchanged.
8. **Verify** merge-vs-adapter numerical equivalence and unloading/rollback before outer evaluation.

Repository-grouped nested cross-validation, five outer folds where feasible (otherwise leave-one-repository-out), is orchestrated by the experiment driver. Do not use held-out data to build or select candidates.

## Input data format

See `examples/` for minimal JSONL and NPZ conventions.

### Trajectory JSONL

One JSON object per task/run:

```json
{
  "task_id": "repo-issue-1",
  "repository": "owner/repo",
  "partition": "diagnostic",
  "resolved": false,
  "output_limit_hit": true,
  "token_ids": [1, 2, 3],
  "events": [
    {"type": "tool_call", "valid": true, "state_changed": false},
    {"type": "test_result", "changed_next_action": true}
  ],
  "spans": {
    "malformed_tool_call": [[12, 19]],
    "looping": [[40, 80]]
  }
}
```

`spans` are token-index intervals, half-open. A production harness adapter should populate them using the pinned tokenizer and released detector. The built-in detector can infer looping from token IDs; malformed calls should be marked by the harness parser or supplied as spans.

### Activation NPZ

For each module, save:
- `task_id`: string array, one entry per row;
- `category`: string array;
- `role`: string array (`failing` or `successful`);
- `progress_bin`: integer array for normalized execution-stage matching;
- `activations`: float32 array `[rows, tokens, d_out]` or `[rows, d_out]`;
- optionally `token_mask`: boolean array for token-level rows.

The extraction code in `student.py` includes a capture helper. For a real run, use a harness adapter to align token positions and save the activations with the metadata above. Avoid storing secrets or private repository content unnecessarily.

## Configuration

See `configs/experiment.yaml`. Key constraints:
- `max_candidates_per_category: 12`
- `max_validation_rollouts_per_method: 300`
- `adapter_footprint_fraction: 0.001`
- `dose_grid: [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]`
- `ranks: [1, 2, 4, 8]`
- `block_range_fraction: [0.35, 0.65]`
- target modules: `o_proj`, `down_proj`
- `alpha = rank`, so effective multiplier is dose.

## Important implementation notes

- **Architecture discovery is mandatory.** Gemma family module paths and implementation details can differ by Transformers version. The workbench discovers matching suffixes and requires explicit confirmation rather than assuming a path.
- **Hook semantics matter.** The paper specifies output of a linear projection, after the linear map and before residual addition or normalization. The hook helper captures module outputs, but verify the exact computation graph for the selected Gemma 4 revision.
- **No autograd training.** Construction uses NumPy/PyTorch tensor algebra only. The model is put in eval mode; gradient tracking is disabled for inference. No optimizer or backward pass is used.
- **Candidate selection.** The frontier model selects only from IDs generated locally. The deterministic baseline uses the same candidate set and budget.
- **Small benchmark.** 129 tasks are a feasibility setting. The code reports estimates and intervals; it does not promise statistical significance.
- **Safety.** This workbench is for controlled research. Do not deploy a patched model unless all pre-registered retention and safety gates pass.

## Current scope / limitations

Included:
- data schemas and failure detectors;
- repository-grouped fold generation;
- activation aggregation and SVD direction construction;
- projection update and LoRA factorization;
- candidate filtering and footprint accounting;
- frontier-model tool protocol and OpenAI-compatible provider;
- fixed and deterministic baselines;
- rollout accounting, acceptance gates, manifest and audit logging;
- numerical adapter equivalence helpers;
- offline demo and unit tests.

Not included (must be implemented for the competition environment):
- a concrete adapter for the official SWE-agent competition harness;
- automatic collection of successful/failing spans from harness events;
- a fully validated Gemma 4 e2B architecture-specific hook map;
- a ready-made provider SDK for every frontier model vendor;
- a statistically powered multi-benchmark study.

## Project structure

```text
LSP_Python_Workbench/
├── README.md
├── pyproject.toml
├── .env.example
├── configs/experiment.yaml
├── examples/
├── lsp/
└── tests/
```
