import importlib.util
import sys
import unittest
from pathlib import Path
from typing import Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODULE_PATH = ROOT / "wan/modules/fademem_memory.py"
SPEC = importlib.util.spec_from_file_location("fademem_memory", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
fademem_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fademem_module
SPEC.loader.exec_module(fademem_module)

ContinuousFadeMem = fademem_module.ContinuousFadeMem
FadeMemConfig = fademem_module.FadeMemConfig


def _make_freqs(max_len: int, head_dim: int) -> torch.Tensor:
    tc = ContinuousFadeMem._temporal_c(head_dim)
    positions = torch.arange(max_len, dtype=torch.float64).unsqueeze(1)
    bands = torch.arange(1, tc + 1, dtype=torch.float64).unsqueeze(0)
    angles = positions * bands * 0.013
    return torch.polar(torch.ones_like(angles), angles)


def _make_fademem(slots: int = 8) -> ContinuousFadeMem:
    cfg = FadeMemConfig(
        enabled=True,
        summary_slots=slots,
    )
    return ContinuousFadeMem(cfg)


def _make_cache(
    mem: ContinuousFadeMem,
    count: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> dict:
    torch.manual_seed(1234 + count)
    cache = mem.init_cache(
        batch_size=2,
        num_heads=2,
        head_dim=8,
        frame_h=2,
        frame_w=3,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    cache["fademem_count"].fill_(count)
    for i in range(count):
        cache["fademem_k"][:, i] = torch.randn_like(cache["fademem_k"][:, i])
        cache["fademem_v"][:, i] = torch.randn_like(cache["fademem_v"][:, i])
        cache["fademem_center_frame"][i] = i * 2
        cache["fademem_kv_frame"][i] = i * 2
        cache["fademem_span"][i] = i + 1
    return cache


def _make_update(
    frame_start: int,
    *,
    requires_grad: bool = False,
    dtype: torch.dtype = torch.float32,
) -> Tuple[dict, torch.Tensor, torch.Tensor]:
    torch.manual_seed(4321 + frame_start)
    k = torch.randn(2, 2, 6, 2, 8, dtype=dtype)
    v = torch.randn(2, 2, 6, 2, 8, dtype=dtype)
    if requires_grad:
        k.requires_grad_()
        v.requires_grad_()
    center_frame = torch.tensor([frame_start, frame_start + 1], dtype=torch.long)
    update = {
        "k": k,
        "v": v,
        "center_frame": center_frame,
        "kv_frame": center_frame.clone(),
        "span": torch.ones(2, dtype=torch.long),
    }
    return update, k, v


def _cat_segments(
    segments: Tuple[Optional[list], Optional[list]],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    k_segments, v_segments = segments
    if k_segments is None or v_segments is None:
        return None, None
    return torch.cat(k_segments, dim=1), torch.cat(v_segments, dim=1)


class FadeMemPlainVectorizedTest(unittest.TestCase):
    def test_released_warp_defaults_to_power_and_rejects_other_types(self):
        default = ContinuousFadeMem(FadeMemConfig())
        self.assertEqual(default.cfg.warp_type, "power")
        with self.assertRaisesRegex(ValueError, "Unknown FadeMem warp type: log"):
            ContinuousFadeMem(FadeMemConfig(warp_type="log"))
        with self.assertRaisesRegex(ValueError, "Unknown FadeMem warp type: unknown"):
            ContinuousFadeMem(FadeMemConfig(warp_type="unknown"))

    def assert_matches_loop(
        self,
        mem: ContinuousFadeMem,
        cache: dict,
        freqs: torch.Tensor,
        *,
        update_info: Optional[dict] = None,
        snapshot: Optional[dict] = None,
    ) -> None:
        entries = mem._simulate_entries(
            cache,
            update_info,
            current_frame=12,
            snapshot=snapshot,
            clone_entries=False,
        )
        loop = mem._build_plain_prefix_segments_loop(
            entries,
            freqs,
        )
        public = mem.build_attention_prefix_segments(
            cache=cache,
            update_info=update_info,
            current_frame=12,
            snapshot=snapshot,
            freqs=freqs,
        )
        loop_k, loop_v = _cat_segments(loop)
        public_k, public_v = _cat_segments(public)
        self.assertIsNotNone(public_k)
        self.assertEqual(len(public[0]), 1)
        torch.testing.assert_close(public_k, loop_k, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(public_v, loop_v, rtol=1e-6, atol=1e-6)

    def test_vectorized_matches_loop_for_snapshot_and_update_cases(self):
        freqs = _make_freqs(max_len=32, head_dim=8)

        for use_snapshot in (False, True):
            for use_update in (False, True):
                with self.subTest(use_snapshot=use_snapshot, use_update=use_update):
                    mem = _make_fademem(slots=8)
                    cache = _make_cache(mem, count=3)
                    snapshot = mem.clone_snapshot(cache) if use_snapshot else None
                    if use_snapshot:
                        cache["fademem_k"][:, :3] += 17.0
                        cache["fademem_center_frame"][:3] += 5
                    update_info = _make_update(7)[0] if use_update else None
                    self.assert_matches_loop(
                        mem,
                        cache,
                        freqs,
                        update_info=update_info,
                        snapshot=snapshot,
                    )

    def test_vectorized_falls_back_when_entry_shapes_differ(self):
        mem = _make_fademem(slots=8)
        freqs = _make_freqs(max_len=32, head_dim=8)
        cache = _make_cache(mem, count=2)
        entries = mem._simulate_entries(cache, None, current_frame=12, clone_entries=False)
        entries[1] = dict(entries[1])
        entries[1]["k"] = entries[1]["k"][:, :-1]
        entries[1]["v"] = entries[1]["v"][:, :-1]
        vectorized = mem._build_plain_prefix_segments_vectorized(
            entries,
            freqs,
        )
        self.assertIsNone(vectorized)

    def test_vectorized_preserves_pending_update_gradients(self):
        freqs = _make_freqs(max_len=32, head_dim=8)

        def run(use_vectorized: bool) -> Tuple[torch.Tensor, torch.Tensor]:
            mem = _make_fademem(slots=8)
            cache = _make_cache(mem, count=2)
            update_info, update_k, update_v = _make_update(7, requires_grad=True)
            with torch.enable_grad():
                entries = mem._simulate_entries(
                    cache,
                    update_info,
                    current_frame=12,
                    clone_entries=False,
                )
                if use_vectorized:
                    segments = mem._build_plain_prefix_segments_vectorized(
                        entries,
                        freqs,
                    )
                else:
                    segments = mem._build_plain_prefix_segments_loop(
                        entries,
                        freqs,
                    )
                out_k, out_v = _cat_segments(segments)
                loss = (out_k.float().square().mean() + out_v.float().square().mean())
                loss.backward()
            return update_k.grad.detach().clone(), update_v.grad.detach().clone()

        loop_k_grad, loop_v_grad = run(use_vectorized=False)
        vector_k_grad, vector_v_grad = run(use_vectorized=True)
        torch.testing.assert_close(vector_k_grad, loop_k_grad, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(vector_v_grad, loop_v_grad, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
