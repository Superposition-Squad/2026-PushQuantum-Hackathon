"""Official validity checks + metrics for a (circuit, loading) pair, as bench/bench.py runs them."""
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "bench"))
logging.getLogger("pushquantum_validator").setLevel(logging.ERROR)

from pushquantum_validator.verify import compute_metrics, strip_moves, verify_equivalence, verify_movement  # noqa: E402
import bench as _bench  # noqa: E402


def check(compiled, loading, reference=None):
    """Returns dict(distance, depth, moves, equivalence). Raises on any violation."""
    movement = verify_movement(compiled, loading, log_file=os.devnull)
    gates = strip_moves(compiled)
    m = compute_metrics(compiled, movement, gates)
    _bench.zonespan_check(movement.trace)
    _bench.physics_check(compiled, loading, movement.trace)
    eq = "not checked"
    if reference is not None:
        r = verify_equivalence(compiled, reference, gates)
        eq = "pairs only (>19q)" if r is None else str(r.equivalence).split(".")[-1]
    return {"distance": m.move_distance, "depth": m.depth, "moves": m.num_moves, "equivalence": eq}
