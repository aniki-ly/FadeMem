"""Interface checks for the fixed released online re-rope layout.

Run:
  python tests/test_online_rerope_layout.py
"""
import os
import sys
from inspect import signature

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.wan_wrapper import WanDiffusionWrapper  # noqa: E402
from wan.modules.causal_model_fademem import CausalWanModel  # noqa: E402
from wan.modules.fademem_memory import (  # noqa: E402
    FADEMEM_ROPE_CHUNK_FRAMES,
    FADEMEM_ROPE_CURRENT_START,
    FADEMEM_ROPE_WINDOW_FRAMES,
    FadeMemConfig,
)


if __name__ == "__main__":
    assert FADEMEM_ROPE_WINDOW_FRAMES == 21
    assert FADEMEM_ROPE_CHUNK_FRAMES == 3
    assert FADEMEM_ROPE_CURRENT_START == 18
    for field in ("rope_mode", "rope_chunk_frames", "warp_tau"):
        assert field not in signature(FadeMemConfig).parameters
    for field in (
        "fademem_rope_mode",
        "fademem_rope_chunk_frames",
        "fademem_warp_tau",
    ):
        assert field not in signature(WanDiffusionWrapper.__init__).parameters
        assert field not in signature(CausalWanModel.__init__).parameters
    for kwargs in (
        {"rope_mode": "off"},
        {"rope_chunk_frames": 3},
        {"warp_tau": 8.0},
    ):
        try:
            FadeMemConfig(**kwargs)
        except TypeError:
            pass
        else:
            raise AssertionError(f"removed configuration was accepted: {kwargs}")
    print("ALL FIXED ONLINE REROPE INTERFACE CHECKS PASSED")
