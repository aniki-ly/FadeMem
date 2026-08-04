"""Boundary-layout checks for the fixed released online re-rope layout.

Run:
  python tests/test_online_rerope_boundary_layout.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wan.modules.causal_model_fademem import CausalWanSelfAttention  # noqa: E402
from wan.modules.fademem_memory import (  # noqa: E402
    ContinuousFadeMem,
    FadeMemConfig,
)


G_VALUES = (0, 3, 12, 15, 18, 21, 120, 240, 360)


def make_entries():
    centers = (0, 6, 12, 21, 36, 54, 78, 108, 144, 192, 252, 318)
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


def online_ready(current_start_frame):
    attn = CausalWanSelfAttention(
        dim=12,
        num_heads=1,
        local_attn_size=3,
        sink_size=0,
        fademem_cfg=make_cfg(),
    )
    kv_cache = {"fademem_count": torch.tensor([12], dtype=torch.long)}
    return attn._online_rerope_attention_ready(kv_cache, None, current_start_frame)


def check_layout():
    mem = ContinuousFadeMem(make_cfg())
    entries = make_entries()
    absolute_positions = [int(e["center_frame"]) for e in entries]

    for count in (3, 6, 9):
        prefull = entries[:count]
        positions = mem._memory_rope_positions(prefull, 120)
        assert positions == [int(e["center_frame"]) for e in prefull], (count, positions)

    for G in G_VALUES:
        positions = mem._memory_rope_positions(entries, G)
        ready = online_ready(G)
        if G < 18:
            assert positions == absolute_positions, (G, positions)
            assert ready is False, (G, ready)
            continue

        assert ready is True, (G, ready)
        assert positions[0] == 0, (G, positions)
        assert len(positions) == len(set(positions)), (G, positions)
        assert not (set(positions) & {18, 19, 20}), (G, positions)

        normal_gaps = [18 - p for p in positions[1:]]
        bridge_gaps = normal_gaps[-3:]
        far_gaps = normal_gaps[:-3]
        assert bridge_gaps == [3, 2, 1], (G, normal_gaps)
        assert all(4 <= gap <= 17 for gap in far_gaps), (G, normal_gaps)
        assert normal_gaps == sorted(normal_gaps, reverse=True), (G, normal_gaps)

    print("fixed released boundary layout passed")


if __name__ == "__main__":
    check_layout()
    print("ALL BOUNDARY ONLINE REROPE LAYOUT CHECKS PASSED")
