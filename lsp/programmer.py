from __future__ import annotations
import json, os, requests
from typing import Protocol
from .schemas import Candidate, Selection
from .manifest import AuditLog

TOOL_SCHEMAS = [
 {"type":"function","function":{"name":"list_candidates","description":"List the fixed candidate pool. No new candidates can be created.","parameters":{"type":"object","properties":{"category":{"type":"string"}},"required":[]}}},
 {"type":"function","function":{"name":"inspect_candidate","description":"Inspect one candidate and its validation summary.","parameters":{"type":"object","properties":{"candidate_id":{"type":"string"}},"required":["candidate_id"]}}},
 {"type":"function","function":{"name":"compare_validation","description":"Compare available candidate validation summaries.","parameters":{"type":"object","properties":{"candidate_ids":{"type":"array","items":{"type":"string"}}},"required":["candidate_ids"]}}},
 {"type":"function","function":{"name":"select_candidates","description":"Propose candidate IDs, ranks and doses. The workbench validates budgets and allowed values.","parameters":{"type":"object","properties":{"candidate_ids":{"type":"array","items":{"type":"string"}},"ranks":{"type":"object","additionalProperties":{"type":"integer"}},"doses":{"type":"object","additionalProperties":{"type":"number"}},"rationale":{"type":"string"}},"required":["candidate_ids"]}}},
 {"type":"function","function":{"name":"finalize","description":"Finish the selection. Empty candidate_ids means return no adapter.","parameters":{"type":"object","properties":{"candidate_ids":{"type":"array","items":{"type":"string"}},"ranks":{"type":"object","additionalProperties":{"type":"integer"}},"doses":{"type":"object","additionalProperties":{"type":"number"}},"rationale":{"type":"string"}},"required":["candidate_ids"]}}}
]

class BaseProgrammer(Protocol):
    def decide(self, brief: dict, tools: list[dict], dispatch): ...

class OpenAICompatibleProgrammer:
    def __init__(self, base_url=None, api_key=None, model=None, temperature=0,
                 max_tool_rounds=12, timeout=180):
        self.base_url = (base_url or os.getenv("LSP_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("LSP_API_KEY")
        self.model = model or os.getenv("LSP_MODEL")
        self.temperature = temperature
        self.max_tool_rounds = max_tool_rounds
        self.timeout = timeout
        if not self.model:
            raise ValueError("Set LSP_MODEL or pass model.")
    def decide(self, brief, tools, dispatch):
        if not self.api_key:
            raise ValueError("Set LSP_API_KEY.")
        messages = [
            {"role":"system","content":
             "You are the frontier-model programmer in an LSP experiment. You may only choose from the fixed candidate pool. Do not request or execute arbitrary code. Respect all validation evidence and budgets. If no candidate meets the preregistered gate, return an empty selection. Explain uncertainty."},
            {"role":"user","content":json.dumps(brief, default=str)}
        ]
        headers = {"Authorization":f"Bearer {self.api_key}", "Content-Type":"application/json"}
        for _ in range(self.max_tool_rounds):
            payload = {"model":self.model,"messages":messages,"tools":tools,
                       "tool_choice":"auto","temperature":self.temperature}
            r = requests.post(self.base_url + "/chat/completions", headers=headers,
                              json=payload, timeout=self.timeout)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            messages.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                return {"selection":None, "transcript":messages}
            for call in calls:
                name = call["function"]["name"]
                args = json.loads(call["function"].get("arguments") or "{}")
                try:
                    result = dispatch(name, args)
                except Exception as e:
                    result = {"error":str(e)}
                messages.append({"role":"tool","tool_call_id":call["id"],
                                 "content":json.dumps(result, default=str)})
                if name == "finalize":
                    return {"selection":result, "transcript":messages}
        raise RuntimeError("Frontier programmer exceeded max_tool_rounds without finalize.")

class ConstrainedWorkbench:
    def __init__(self, candidates, validation_summaries, allowed_ranks=(1,2,4,8),
                 allowed_doses=(0,.25,.5,.75,1,1.25,1.5), footprint_cap=None,
                 audit_log=None):
        self.candidates = {c.candidate_id:c for c in candidates}
        self.validation = validation_summaries
        self.allowed_ranks, self.allowed_doses = set(allowed_ranks), set(allowed_doses)
        self.footprint_cap = footprint_cap
        self.audit = AuditLog(audit_log) if audit_log else None
        self.selection = None
    def dispatch(self, name, args):
        if name == "list_candidates":
            cat = args.get("category")
            rows = [c.model_dump() for c in self.candidates.values()
                    if cat is None or c.category == cat]
            return {"candidates":rows}
        if name == "inspect_candidate":
            cid = args["candidate_id"]
            if cid not in self.candidates: raise KeyError(cid)
            return {"candidate":self.candidates[cid].model_dump(),
                    "validation":self.validation.get(cid)}
        if name == "compare_validation":
            ids = args["candidate_ids"]
            if any(i not in self.candidates for i in ids): raise KeyError("Unknown candidate ID")
            return {i:self.validation.get(i) for i in ids}
        if name in {"select_candidates","finalize"}:
            selection = self._validate_selection(args)
            self.selection = selection
            if self.audit: self.audit.write("frontier_selection", **selection)
            return selection
        raise ValueError(f"Tool not allowed: {name}")
    def _validate_selection(self, args):
        ids = list(dict.fromkeys(args.get("candidate_ids", [])))
        ranks, doses = args.get("ranks", {}), args.get("doses", {})
        total = 0
        normalized = []
        for cid in ids:
            if cid not in self.candidates: raise ValueError(f"Unknown candidate: {cid}")
            c = self.candidates[cid]
            rank = int(ranks.get(cid, 1))
            dose = float(doses.get(cid, 1.0))
            if rank not in self.allowed_ranks: raise ValueError(f"Disallowed rank: {rank}")
            if dose not in self.allowed_doses: raise ValueError(f"Disallowed dose: {dose}")
            total += rank * (c.d_in + c.d_out)
            normalized.append({"candidate_id":cid,"rank":rank,"dose":dose})
        if self.footprint_cap is not None and total > self.footprint_cap:
            raise ValueError(f"Footprint {total} exceeds cap {self.footprint_cap}")
        return {"candidate_ids":ids,"items":normalized,
                "footprint":total,"rationale":args.get("rationale","")}
