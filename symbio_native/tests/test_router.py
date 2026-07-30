from symbio_native.router import Router
from symbio_native.substrate import Branch


def test_router_selects_by_trigger_overlap():
    b1 = Branch("math", trigger_ids=[1, 2, 3], weights={}, description="math branch")
    b2 = Branch("greeting", trigger_ids=[10, 11], weights={}, description="greeting branch")
    router = Router([b1, b2])
    selected = router.select([2, 3, 99])
    assert len(selected) == 1
    assert selected[0].name == "math"


def test_router_top_k():
    branches = [
        Branch(f"b{i}", trigger_ids=[i], weights={}, description=str(i))
        for i in range(5)
    ]
    router = Router(branches)
    selected = router.select([0, 1, 2, 3], top_k=2)
    assert len(selected) == 2
