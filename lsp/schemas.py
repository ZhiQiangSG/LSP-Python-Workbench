from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field, ConfigDict

FailureCategory = Literal["malformed_tool_call", "looping", "token_exhaustion"]

class Event(BaseModel):
    type: str
    valid: bool | None = None
    state_changed: bool | None = None
    changed_next_action: bool | None = None
    successful_termination: bool | None = None

class Trajectory(BaseModel):
    model_config = ConfigDict(extra="allow")
    task_id: str
    repository: str
    partition: str
    resolved: bool = False
    output_limit_hit: bool = False
    token_ids: list[int] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    spans: dict[str, list[tuple[int, int]]] = Field(default_factory=dict)
    failures: list[FailureCategory] = Field(default_factory=list)

class Candidate(BaseModel):
    candidate_id: str
    category: FailureCategory
    module_name: str
    block_index: int
    sign: int
    rank: int = 1
    dose: float = 1.0
    diagnostic_score: float = 0.0
    d_in: int
    d_out: int
    footprint: int
    singular_values: list[float] = Field(default_factory=list)
    factors_path: str | None = None

class Selection(BaseModel):
    candidate_ids: list[str]
    ranks: dict[str, int] = Field(default_factory=dict)
    doses: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""

class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]
