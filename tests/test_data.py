from lsp.data import detect_repeated_8grams, repository_grouped_folds
from lsp.schemas import Trajectory

def test_detect_repeated_8gram():
    g=[1,2,3,4,5,6,7,8]
    assert detect_repeated_8grams(g*3)

def test_repository_grouped_folds_no_leakage():
    rows=[Trajectory(task_id=f"{r}-{i}",repository=r,partition="x")
          for r in ["a","b","c","d"] for i in range(2)]
    folds=repository_grouped_folds(rows,2,0)
    for f in folds:
        train={x.repository for x in f["train"]}
        test={x.repository for x in f["test"]}
        assert not train & test
