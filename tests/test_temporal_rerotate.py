"""Numerical and cache-write checks for 6.25 online temporal re-rope.

Run:
  python tests/test_temporal_rerotate.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wan.modules.causal_model_fademem as causal_mod  # noqa: E402
from wan.modules.causal_model_fademem import (  # noqa: E402
    CausalWanSelfAttention,
    causal_rope_apply,
    rerotate_temporal_rope,
)
from wan.modules.model import rope_params  # noqa: E402
from wan.modules.fademem_memory import (  # noqa: E402
    ContinuousFadeMem,
    FadeMemConfig,
)


def make_freqs(max_seq_len=1024, head_dim=12):
    return torch.cat(
        [
            rope_params(max_seq_len, head_dim - 4 * (head_dim // 6)),
            rope_params(max_seq_len, 2 * (head_dim // 6)),
            rope_params(max_seq_len, 2 * (head_dim // 6)),
        ],
        dim=1,
    )


def check_rerotate(old_start, target_start):
    torch.manual_seed(7 + old_start + target_start)
    batch, frames, height, width, heads, head_dim = 2, 3, 2, 2, 2, 12
    frame_seqlen = height * width
    raw = torch.randn(batch, frames * frame_seqlen, heads, head_dim, dtype=torch.float32)
    grid_sizes = torch.tensor([[frames, height, width]] * batch, dtype=torch.long)
    freqs = make_freqs(head_dim=head_dim)

    old_frames = torch.arange(old_start, old_start + frames, dtype=torch.long)
    target_frames = torch.arange(target_start, target_start + frames, dtype=torch.long)
    old_roped = causal_rope_apply(raw, grid_sizes, freqs, start_frame=old_start)
    expected = causal_rope_apply(raw, grid_sizes, freqs, start_frame=target_start)
    actual = rerotate_temporal_rope(
        old_roped,
        freqs,
        old_frames,
        target_frames,
        frame_seqlen,
    )
    err = (actual - expected).abs().max().item()
    assert err < 1e-4, (old_start, target_start, err)
    print(f"rerotate old={old_start} -> target={target_start}: max_abs={err:.3e}")


def check_unrotate_temporal():
    torch.manual_seed(23)
    batch, frames, height, width, heads, head_dim = 1, 1, 1, 1, 2, 12
    raw = torch.randn(batch, frames * height * width, heads, head_dim, dtype=torch.float32)
    grid_sizes = torch.tensor([[frames, height, width]], dtype=torch.long)
    freqs = make_freqs(head_dim=head_dim)
    old_frame = 240
    roped = causal_rope_apply(raw, grid_sizes, freqs, start_frame=old_frame)
    mem = ContinuousFadeMem(FadeMemConfig(enabled=True))
    actual = mem._un_rotate_temporal(roped, freqs, old_frame)
    err = (actual - raw).abs().max().item()
    assert err < 1e-4, err
    print(f"summary unrotate old={old_frame}: max_abs={err:.3e}")


def check_cache_update_new_k_is_absolute(target_start):
    saved_attention = causal_mod.attention
    captured = {}

    def fake_attention(q, k, v):
        captured["q"] = q.detach().clone()
        captured["k"] = k.detach().clone()
        return torch.zeros_like(q)

    causal_mod.attention = fake_attention
    try:
        torch.manual_seed(31)
        dim, heads, frames, frame_seqlen = 12, 1, 3, 1
        cfg = FadeMemConfig(
            enabled=True,
            summary_slots=12,
            warp_type="power",
            warp_beta=0.3,
        )
        attn = CausalWanSelfAttention(
            dim=dim,
            num_heads=heads,
            local_attn_size=3,
            sink_size=0,
            fademem_cfg=cfg,
        )
        x = torch.randn(1, frames * frame_seqlen, dim)
        grid_sizes = torch.tensor([[frames, 1, 1]], dtype=torch.long)
        seq_lens = torch.tensor([frames * frame_seqlen], dtype=torch.long)
        freqs = make_freqs(head_dim=dim)
        current_start_frame = 120
        kv_cache = {
            "k": torch.zeros(1, frames * frame_seqlen, heads, dim),
            "v": torch.zeros(1, frames * frame_seqlen, heads, dim),
            "global_end_index": torch.tensor([current_start_frame * frame_seqlen], dtype=torch.long),
            "local_end_index": torch.tensor([0], dtype=torch.long),
            "fademem_k": torch.zeros(1, 12, 1, heads, dim),
            "fademem_v": torch.zeros(1, 12, 1, heads, dim),
            "fademem_center_frame": torch.arange(12, dtype=torch.long),
            "fademem_kv_frame": torch.arange(12, dtype=torch.long),
            "fademem_span": torch.ones(12, dtype=torch.long),
            "fademem_count": torch.tensor([12], dtype=torch.long),
        }

        with torch.no_grad():
            _, (_, _, update_info) = attn(
                x,
                seq_lens,
                grid_sizes,
                freqs,
                block_mask=None,
                kv_cache=kv_cache,
                current_start=current_start_frame * frame_seqlen,
            )
            raw_k = attn.norm_k(attn.k(x)).view(1, frames * frame_seqlen, heads, dim)
            expected_abs = causal_rope_apply(
                raw_k,
                grid_sizes,
                freqs,
                start_frame=current_start_frame,
            )
            attention_roped = causal_rope_apply(
                raw_k,
                grid_sizes,
                freqs,
                start_frame=target_start,
            )

        new_k = update_info["new_k"]
        abs_err = (new_k - expected_abs).abs().max().item()
        attn_err = (new_k - attention_roped).abs().max().item()
        assert abs_err < 1e-4, abs_err
        assert attn_err > 1e-3, attn_err
        attn_k = captured["k"][:, -frames * frame_seqlen:]
        attn_view_err = (attn_k - attention_roped).abs().max().item()
        attn_abs_diff = (attn_k - expected_abs).abs().max().item()
        assert attn_view_err < 1e-4, attn_view_err
        assert attn_abs_diff > 1e-3, attn_abs_diff
        print(
            f"fixed layout: cache_update_info['new_k'] uses absolute RoPE: "
            f"abs_err={abs_err:.3e}, attn_err={attn_err:.3e}, "
            f"attn_view_err={attn_view_err:.3e}"
        )
    finally:
        causal_mod.attention = saved_attention


if __name__ == "__main__":
    for old_start, target_start in (
        (0, 12),
        (120, 12),
        (240, 18),
        (480, 18),
    ):
        check_rerotate(old_start, target_start)
    check_unrotate_temporal()
    check_cache_update_new_k_is_absolute(18)
    print("ALL TEMPORAL REROTATE CHECKS PASSED")
