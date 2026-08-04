"""Far-memory allocator checks for the released online re-rope mode.

Run:
  python tests/test_online_rerope_far_allocator.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wan.modules.fademem_memory import (  # noqa: E402
    ContinuousFadeMem,
    FadeMemConfig,
)


def make_entries():
    centers = (0, 10, 20, 40, 80, 120, 160, 200, 240, 280, 320, 340)
    return [
        {"center_frame": c, "kv_frame": c, "span": 1}
        for c in centers
    ]


def make_cfg():
    return FadeMemConfig(
        enabled=True,
        summary_slots=12,
        warp_type="power",
        warp_beta=0.3,
        anchor_slots=1,
    )


def positions_and_gaps(G=360):
    mem = ContinuousFadeMem(make_cfg())
    positions = mem._memory_rope_positions(make_entries(), G)
    normal_gaps = [18 - p for p in positions[1:]]
    return positions, normal_gaps[:-3], normal_gaps[-3:]


def check_common_far_invariants():
    positions, far_gaps, bridge_gaps = positions_and_gaps()
    assert positions[0] == 0, positions
    assert len(positions) == len(set(positions)), positions
    assert not (set(positions) & {18, 19, 20}), positions
    assert bridge_gaps == [3, 2, 1], bridge_gaps
    assert all(4 <= gap <= 17 for gap in far_gaps), far_gaps
    assert far_gaps == sorted(far_gaps, reverse=True), far_gaps
    assert len(far_gaps) == len(set(far_gaps)), far_gaps
    return far_gaps


if __name__ == "__main__":
    check_common_far_invariants()
    print("ALL FAR ALLOCATOR CHECKS PASSED")
