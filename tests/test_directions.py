import numpy as np
from lsp.directions import projection_update, factorize_update, lora_dense_delta

def test_rank_one_factorization_reconstructs_rank_one_update():
    rng=np.random.default_rng(1)
    w=rng.normal(size=(12,9)).astype("float32")
    p,_=np.linalg.qr(rng.normal(size=(12,1)))
    delta=projection_update(w,p.astype("float32"),0.75)
    a,b,err=factorize_update(delta,1,alpha=1,dose=0.75)
    got=lora_dense_delta(a,b,1,1,0.75)
    assert err < 1e-5
    assert np.max(np.abs(delta-got)) < 1e-5

def test_footprint():
    from lsp.directions import footprint
    assert footprint(2,10,20)==60
