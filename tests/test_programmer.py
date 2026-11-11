import pytest
from lsp.schemas import Candidate
from lsp.programmer import ConstrainedWorkbench

def candidate(cid):
    return Candidate(candidate_id=cid,category="looping",module_name="layers.4.o_proj",
                     block_index=4,sign=-1,d_in=8,d_out=8,footprint=16)

def test_programmer_cannot_invent_candidates_or_exceed_budget():
    wb=ConstrainedWorkbench([candidate("abc")],{},footprint_cap=100)
    with pytest.raises(ValueError):
        wb.dispatch("finalize",{"candidate_ids":["not-real"]})
    with pytest.raises(ValueError):
        wb.dispatch("finalize",{"candidate_ids":["abc"],"ranks":{"abc":8}})
