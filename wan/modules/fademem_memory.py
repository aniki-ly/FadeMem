# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


FADEMEM_ROPE_WINDOW_FRAMES = 21
FADEMEM_ROPE_CHUNK_FRAMES = 3
FADEMEM_ROPE_CURRENT_START = FADEMEM_ROPE_WINDOW_FRAMES - FADEMEM_ROPE_CHUNK_FRAMES


@dataclass
class FadeMemConfig:
    """Configuration for the released FadeMem memory path."""

    enabled: bool = True
    summary_slots: int = 8

    # Unified-schedule boundary conditions, applied in a single place
    # (anchor short-circuit in ``_insert_entry``; ``_choose_merge_index``
    # for newest):
    #
    # ``anchor_slots=1``   - frame 0 fully participates in the merge argmin;
    #                         when chosen, the anchor short-circuit keeps its
    #                         KV / center / kv_frame and only accumulates span.
    #                         Set to 0 to disable anchor entirely.
    # ``protect_newest``   - exclude the rightmost gap from the argmin so the
    #                         just-inserted slot has one step to settle.
    anchor_slots: int = 1
    protect_newest: bool = True

    warp_type: str = "power"
    warp_beta: float = 1.0
    span_gamma: float = 0.5


class ContinuousFadeMem:
    """Streaming fixed-budget FadeMem.

    This module is intentionally lightweight and online:
      - exact recent memory remains in the rolling KV cache;
      - evicted frames are converted to persistent summary entries;
      - the far-history bank keeps a fixed number of slots;
      - when full, it merges the adjacent pair that is closest under the
        released power-law age warp.

    Evicted frames retain their full spatial token grid. Adjacent summaries are
    combined with the released effective-span weighted merge.
    """

    def __init__(self, cfg: FadeMemConfig):
        self.cfg = cfg
        self.cfg.warp_type = str(self.cfg.warp_type).lower()
        if self.cfg.warp_type != "power":
            raise ValueError(f"Unknown FadeMem warp type: {self.cfg.warp_type}")
        self.cfg.anchor_slots = max(0, int(getattr(self.cfg, "anchor_slots", 0)))
        self.cfg.protect_newest = bool(getattr(self.cfg, "protect_newest", False))

    # ------------------------------------------------------------------
    # Temporal RoPE helpers (pre-RoPE storage for correct merging)
    # ------------------------------------------------------------------
    @staticmethod
    def _temporal_c(head_dim: int) -> int:
        """Number of complex dims in the temporal RoPE band."""
        c = head_dim // 2
        return c - 2 * (c // 3)

    @staticmethod
    def _un_rotate_temporal(
        k: torch.Tensor,
        freqs: torch.Tensor,
        frame_pos: int,
    ) -> torch.Tensor:
        """Remove temporal RoPE from a single-frame key tensor.

        All S tokens in the tensor share the same temporal position.
        k: [B, S, N, D]     freqs: [max_len, c] complex
        """
        D = k.shape[-1]
        tc = ContinuousFadeMem._temporal_c(D)
        fp = min(frame_pos, freqs.shape[0] - 1)
        k_complex = torch.view_as_complex(
            k.to(torch.float64).contiguous().reshape(*k.shape[:-1], D // 2, 2)
        )
        conj_rot = freqs[fp, :tc].conj().to(k_complex.device)
        k_complex[..., :tc] = k_complex[..., :tc] * conj_rot
        return torch.view_as_real(k_complex).flatten(-2).type_as(k)

    def rope_chunk_base(self, current_start_frame: int) -> int:
        """RoPE position of the current chunk's first frame.

        The current chunk (the 3 denoised frames) ALWAYS keeps its true ABSOLUTE
        position, exactly like the original FadeMem. Successive chunks therefore
        sit at consecutive positions (chunk t at [3t,3t+2], chunk t+1 next), so
        the autoregressive continuation stays smooth. Pinning the chunk to a
        fixed window slot (reset) makes every chunk regenerate at the same slot
        and breaks continuity -> per-chunk flicker. RoPE is relative, so the
        growing absolute index is harmless (only gaps matter, here always <= W).
        """
        return int(current_start_frame)

    def rope_current_target_start(self) -> int:
        """Fixed current-chunk target start for the released online layout."""
        return FADEMEM_ROPE_CURRENT_START

    def _online_rerope_memory_positions(self, entries, current_start_frame: int):
        positions: List[Optional[int]] = [None] * len(entries)
        anchors: List[Tuple[int, Dict[str, torch.Tensor]]] = []
        normals: List[Tuple[int, Dict[str, torch.Tensor]]] = []
        for i, e in enumerate(entries):
            if self._is_anchor_entry(e) and len(anchors) == 0:
                anchors.append((i, e))
            else:
                normals.append((i, e))

        current_pos = FADEMEM_ROPE_CURRENT_START
        bridge_frames = FADEMEM_ROPE_CHUNK_FRAMES
        far_min_gap = bridge_frames + 1
        max_gap = current_pos - 1
        for i, _ in anchors:
            positions[i] = 0
        sorted_normals = sorted(normals, key=lambda item: int(item[1]["center_frame"]))
        bridge = sorted_normals[-min(bridge_frames, len(sorted_normals)):]
        far_normals = sorted_normals[:len(sorted_normals) - len(bridge)]

        for offset, (i, _) in enumerate(bridge):
            gap = len(bridge) - offset
            positions[i] = current_pos - gap

        if len(far_normals) > (max_gap - far_min_gap + 1):
            raise ValueError(
                f"released online re-rope layout has {len(far_normals)} far normal entries but only "
                f"{max_gap - far_min_gap + 1} unique gaps"
            )
        beta = float(getattr(self.cfg, "warp_beta", 0.3))
        G = int(current_start_frame)
        prev_gap = max_gap + 1
        for idx, (i, e) in enumerate(far_normals):
            remaining = len(far_normals) - idx - 1
            lo = far_min_gap + remaining
            hi = prev_gap - 1
            if lo > hi:
                raise ValueError(
                    f"Invalid released online re-rope gap range: lo={lo}, hi={hi}"
                )
            true_gap = max(1, G - int(e["center_frame"]))
            target_gap = min(max(true_gap, far_min_gap), max_gap)
            target_warp = float(target_gap) ** beta
            gap = min(
                range(lo, hi + 1),
                key=lambda g: abs((float(g) ** beta) - target_warp),
            )
            positions[i] = current_pos - gap
            prev_gap = gap

        for i, p in enumerate(positions):
            if p is None:
                raise RuntimeError(f"Failed to assign online re-rope position for entry {i}")
        return [int(p) for p in positions]

    def _memory_rope_positions(self, entries, current_start_frame: int):
        """RoPE temporal position for each memory entry (oldest->newest).

        Disabled FadeMem serves no prefix. Enabled FadeMem keeps absolute
        positions until the summary bank and boundary warmup are ready, then
        assigns the released fixed-layout positions.
        """
        if not self.cfg.enabled or len(entries) == 0:
            return [int(e.get("center_frame")) for e in entries]
        target_start = self.rope_current_target_start()
        if int(current_start_frame) < int(target_start):
            return [int(e.get("center_frame")) for e in entries]
        if len(entries) < int(self.cfg.summary_slots):
            return [int(e.get("center_frame")) for e in entries]
        return self._online_rerope_memory_positions(entries, current_start_frame)

    # ------------------------------------------------------------------
    # Cache allocation / initialization
    # ------------------------------------------------------------------
    def ensure_cache(
        self,
        cache: Dict[str, torch.Tensor],
        batch_size: int,
        num_heads: int,
        head_dim: int,
        frame_h: int,
        frame_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if (not self.cfg.enabled) or ("fademem_k" in cache):
            return
        cache.update(
            self.init_cache(
                batch_size=batch_size,
                num_heads=num_heads,
                head_dim=head_dim,
                frame_h=frame_h,
                frame_w=frame_w,
                device=device,
                dtype=dtype,
            )
        )

    def init_cache(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        frame_h: int,
        frame_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        slot_tokens = int(frame_h) * int(frame_w)
        slots = int(self.cfg.summary_slots)
        return {
            "fademem_k": torch.zeros(
                batch_size, slots, slot_tokens, num_heads, head_dim,
                device=device, dtype=dtype,
            ),
            "fademem_v": torch.zeros(
                batch_size, slots, slot_tokens, num_heads, head_dim,
                device=device, dtype=dtype,
            ),
            "fademem_center_frame": torch.full((slots,), -1, device=device, dtype=torch.long),
            "fademem_kv_frame": torch.full((slots,), -1, device=device, dtype=torch.long),
            "fademem_span": torch.zeros((slots,), device=device, dtype=torch.long),
            "fademem_count": torch.zeros((1,), device=device, dtype=torch.long),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compress_evicted_tokens(
        self,
        evicted_k: Optional[torch.Tensor],
        evicted_v: Optional[torch.Tensor],
        frame_h: int,
        frame_w: int,
        start_frame: int,
        freqs: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if (
            not self.cfg.enabled
            or evicted_k is None
            or evicted_v is None
            or evicted_k.numel() == 0
            or evicted_v.numel() == 0
        ):
            return None

        frame_tokens = frame_h * frame_w
        usable_tokens = (evicted_k.shape[1] // frame_tokens) * frame_tokens
        if usable_tokens <= 0:
            return None

        evicted_k = evicted_k[:, :usable_tokens]
        evicted_v = evicted_v[:, :usable_tokens]
        num_frames = usable_tokens // frame_tokens

        k_frames = evicted_k.view(
            evicted_k.shape[0], num_frames, frame_h, frame_w, evicted_k.shape[2], evicted_k.shape[3]
        )
        v_frames = evicted_v.view(
            evicted_v.shape[0], num_frames, frame_h, frame_w, evicted_v.shape[2], evicted_v.shape[3]
        )

        compressed_k = []
        compressed_v = []
        for i in range(num_frames):
            ck = self._compress_frame(k_frames[:, i], frame_h, frame_w)
            cv = self._compress_frame(v_frames[:, i], frame_h, frame_w)
            if freqs is not None:
                ck = self._un_rotate_temporal(ck, freqs, start_frame + i)
            compressed_k.append(ck)
            compressed_v.append(cv)

        center_frame = torch.arange(
            start_frame,
            start_frame + num_frames,
            device=evicted_k.device,
            dtype=torch.long,
        )
        span = torch.ones_like(center_frame)

        return {
            "k": torch.stack(compressed_k, dim=1),
            "v": torch.stack(compressed_v, dim=1),
            "center_frame": center_frame,
            "kv_frame": center_frame.clone(),
            "span": span,
        }

    def build_attention_prefix(
        self,
        cache: Dict[str, torch.Tensor],
        update_info: Optional[Dict[str, torch.Tensor]],
        current_frame: int,
        max_tokens: Optional[int] = None,
        snapshot: Optional[Dict[str, torch.Tensor]] = None,
        freqs: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        result_k, result_v = self.build_attention_prefix_segments(
            cache=cache,
            update_info=update_info,
            current_frame=current_frame,
            max_tokens=max_tokens,
            snapshot=snapshot,
            freqs=freqs,
        )
        if result_k is None or result_v is None:
            return None, None
        return torch.cat(result_k, dim=1), torch.cat(result_v, dim=1)

    def build_attention_prefix_segments(
        self,
        cache: Dict[str, torch.Tensor],
        update_info: Optional[Dict[str, torch.Tensor]],
        current_frame: int,
        max_tokens: Optional[int] = None,
        snapshot: Optional[Dict[str, torch.Tensor]] = None,
        freqs: Optional[torch.Tensor] = None,
        query_frame: Optional[int] = None,
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        if not self.cfg.enabled or "fademem_k" not in cache:
            return None, None

        entries = self._simulate_entries(
            cache,
            update_info,
            current_frame,
            snapshot=snapshot,
            clone_entries=False,
        )
        if len(entries) == 0:
            return None, None
        if max_tokens is not None:
            entries = self._take_newest_that_fit(entries, max_tokens)
            if len(entries) == 0:
                return None, None

        # 6.22 RoPE schemes: precompute a position per entry (oldest->newest)
        qf = current_frame if query_frame is None else int(query_frame)
        rope_positions = self._memory_rope_positions(entries, qf)

        return self._build_plain_prefix_segments(entries, freqs, rope_positions)

    def _build_plain_prefix_segments(
        self,
        entries: List[Dict[str, torch.Tensor]],
        freqs: Optional[torch.Tensor],
        rope_positions: Optional[List[int]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if freqs is not None:
            vectorized = self._build_plain_prefix_segments_vectorized(entries, freqs, rope_positions)
            if vectorized is not None:
                return vectorized
        return self._build_plain_prefix_segments_loop(entries, freqs, rope_positions)

    def _build_plain_prefix_segments_loop(
        self,
        entries: List[Dict[str, torch.Tensor]],
        freqs: Optional[torch.Tensor],
        rope_positions: Optional[List[int]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        result_k = []
        result_v = []
        for i, e in enumerate(entries):
            k = e["k"]
            if freqs is not None:
                if rope_positions is not None:
                    rope_frame = rope_positions[i]
                else:
                    rope_frame = int(e["center_frame"])
                k = self._apply_temporal_rope_plain(k, freqs, rope_frame)
            result_k.append(k)
            result_v.append(e["v"])
        return result_k, result_v

    def _build_plain_prefix_segments_vectorized(
        self,
        entries: List[Dict[str, torch.Tensor]],
        freqs: torch.Tensor,
        rope_positions: Optional[List[int]] = None,
    ) -> Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        if len(entries) < 2:
            return None

        first_k = entries[0]["k"]
        first_v = entries[0]["v"]
        if first_k.ndim != 4 or first_v.ndim != 4:
            return None

        k_shape = first_k.shape
        v_shape = first_v.shape
        k_device = first_k.device
        v_device = first_v.device
        k_dtype = first_k.dtype
        v_dtype = first_v.dtype
        for e in entries:
            k = e["k"]
            v = e["v"]
            if (
                k.shape != k_shape
                or v.shape != v_shape
                or k.device != k_device
                or v.device != v_device
                or k.dtype != k_dtype
                or v.dtype != v_dtype
            ):
                return None

        k_bank = torch.stack([e["k"] for e in entries], dim=1)
        b, entry_count, slot_tokens, num_heads, head_dim = k_bank.shape
        tc = self._temporal_c(head_dim)
        if tc > 0:
            if rope_positions is not None:
                frame_positions = [int(p) for p in rope_positions]
            else:
                frame_positions = [int(e["center_frame"]) for e in entries]
            fp = torch.as_tensor(frame_positions, device=freqs.device, dtype=torch.long)
            fp = torch.clamp(fp, max=freqs.shape[0] - 1)
            k_complex = torch.view_as_complex(
                k_bank.to(torch.float64).contiguous().reshape(*k_bank.shape[:-1], head_dim // 2, 2)
            )
            rot = freqs[fp, :tc].to(k_complex.device).view(1, entry_count, 1, 1, tc)
            k_complex[..., :tc] = k_complex[..., :tc] * rot
            k_bank = torch.view_as_real(k_complex).flatten(-2).type_as(k_bank)

        return [
            k_bank.reshape(b, entry_count * slot_tokens, num_heads, head_dim)
        ], [e["v"] for e in entries]

    def _apply_temporal_rope_plain(
        self,
        k: torch.Tensor,
        freqs: torch.Tensor,
        frame_pos: int,
    ) -> torch.Tensor:
        D = k.shape[-1]
        tc = self._temporal_c(D)
        fp = min(frame_pos, freqs.shape[0] - 1)
        k_complex = torch.view_as_complex(
            k.to(torch.float64).contiguous().reshape(*k.shape[:-1], D // 2, 2)
        )
        rot = freqs[fp, :tc].to(k_complex.device)
        k_complex[..., :tc] = k_complex[..., :tc] * rot
        return torch.view_as_real(k_complex).flatten(-2).type_as(k)

    def apply_update(
        self,
        cache: Dict[str, torch.Tensor],
        update_info: Optional[Dict[str, torch.Tensor]],
        current_frame: int,
    ) -> None:
        if not self.cfg.enabled or "fademem_k" not in cache:
            return

        entries = self._simulate_entries(cache, update_info, current_frame)
        count = len(entries)

        cache["fademem_count"].fill_(count)

        has_kv_frame = "fademem_kv_frame" in cache
        if count == 0:
            cache["fademem_center_frame"].fill_(-1)
            if has_kv_frame:
                cache["fademem_kv_frame"].fill_(-1)
            cache["fademem_span"].zero_()
            cache["fademem_k"].zero_()
            cache["fademem_v"].zero_()
            return

        for idx, entry in enumerate(entries):
            cache["fademem_k"][:, idx] = entry["k"]
            cache["fademem_v"][:, idx] = entry["v"]
            cache["fademem_center_frame"][idx] = int(entry["center_frame"])
            if has_kv_frame:
                cache["fademem_kv_frame"][idx] = int(entry.get("kv_frame", entry["center_frame"]))
            cache["fademem_span"][idx] = int(entry["span"])

        max_slots = cache["fademem_k"].shape[1]
        if count < max_slots:
            cache["fademem_k"][:, count:] = 0
            cache["fademem_v"][:, count:] = 0
            cache["fademem_center_frame"][count:] = -1
            if has_kv_frame:
                cache["fademem_kv_frame"][count:] = -1
            cache["fademem_span"][count:] = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compress_frame(self, frame: torch.Tensor, frame_h: int, frame_w: int) -> torch.Tensor:
        """Flatten one frame while retaining every spatial token.

        frame: [B, H, W, num_heads, head_dim]
        returns: [B, S_summary, num_heads, head_dim]
        """
        b, h, w, n, d = frame.shape
        assert h == frame_h and w == frame_w
        return frame.reshape(b, frame_h * frame_w, n, d)

    def clone_snapshot(self, cache: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
        if (not self.cfg.enabled) or ("fademem_k" not in cache):
            return None
        snap = {
            "fademem_k": cache["fademem_k"].detach().clone(),
            "fademem_v": cache["fademem_v"].detach().clone(),
            "fademem_center_frame": cache["fademem_center_frame"].detach().clone(),
            "fademem_span": cache["fademem_span"].detach().clone(),
            "fademem_count": cache["fademem_count"].detach().clone(),
        }
        if "fademem_kv_frame" in cache:
            snap["fademem_kv_frame"] = cache["fademem_kv_frame"].detach().clone()
        return snap

    def _load_entries(
        self,
        cache: Dict[str, torch.Tensor],
        clone_tensors: bool = True,
    ) -> List[Dict[str, torch.Tensor]]:
        count = int(cache["fademem_count"].item())
        has_kv_frame = "fademem_kv_frame" in cache
        entries: List[Dict[str, torch.Tensor]] = []
        for i in range(count):
            cf = int(cache["fademem_center_frame"][i].item())
            k = cache["fademem_k"][:, i]
            v = cache["fademem_v"][:, i]
            if clone_tensors:
                k = k.clone()
                v = v.clone()
            entries.append(
                {
                    "k": k,
                    "v": v,
                    "center_frame": cf,
                    "kv_frame": int(cache["fademem_kv_frame"][i].item()) if has_kv_frame else cf,
                    "span": int(cache["fademem_span"][i].item()),
                }
            )
        return entries

    def _simulate_entries(
        self,
        cache: Dict[str, torch.Tensor],
        update_info: Optional[Dict[str, torch.Tensor]],
        current_frame: int,
        snapshot: Optional[Dict[str, torch.Tensor]] = None,
        clone_entries: bool = True,
    ) -> List[Dict[str, torch.Tensor]]:
        source_cache = cache if snapshot is None else snapshot
        entries = self._load_entries(source_cache, clone_tensors=clone_entries)
        if update_info is None:
            return entries

        if bool(update_info.get("flush_before", False)):
            entries = []

        if "center_frame" not in update_info:
            return entries

        new_count = int(update_info["center_frame"].shape[0])
        if new_count == 0:
            return entries

        has_kv_frame = "kv_frame" in update_info
        for i in range(new_count):
            cf = int(update_info["center_frame"][i].item())
            new_entry = {
                "k": update_info["k"][:, i],
                "v": update_info["v"][:, i],
                "center_frame": cf,
                "kv_frame": int(update_info["kv_frame"][i].item()) if has_kv_frame else cf,
                "span": int(update_info["span"][i].item()),
            }
            entries = self._insert_entry(entries, new_entry, current_frame=current_frame)
        return entries

    def _insert_entry(
        self,
        entries: List[Dict[str, torch.Tensor]],
        new_entry: Dict[str, torch.Tensor],
        current_frame: int,
    ) -> List[Dict[str, torch.Tensor]]:
        max_slots = int(self.cfg.summary_slots)
        if len(entries) < max_slots:
            return entries + [new_entry]

        work = entries + [new_entry]
        merge_idx = self._choose_merge_index(work, current_frame=current_frame)

        # Anchor short-circuit: if either side is the anchor, keep its KV
        # verbatim; only ``span`` accumulates.
        # This makes the anchor a true schedule participant rather than a
        # pinned outsider, while still guaranteeing frame 0 is never lost.
        a_pair, b_pair = work[merge_idx], work[merge_idx + 1]
        if self._is_anchor_entry(a_pair) or self._is_anchor_entry(b_pair):
            kept = a_pair if self._is_anchor_entry(a_pair) else b_pair
            compacted = {
                "k": kept["k"], "v": kept["v"],
                "center_frame": 0, "kv_frame": 0,
                "span": int(a_pair["span"]) + int(b_pair["span"]),
            }
            if _log:
                print(f"[FadeMem]     -> anchor: center=0, kv_frame=0, span={int(compacted['span'])}")
            out: List[Dict[str, torch.Tensor]] = []
            out.extend(work[:merge_idx])
            out.append(compacted)
            out.extend(work[merge_idx + 2:])
            assert len(out) == max_slots
            return out

        merged = self._merge_two_entries(
            work[merge_idx],
            work[merge_idx + 1],
        )
        if _log:
            print(f"[FadeMem]     -> merge: center={int(merged['center_frame'])}, span={int(merged['span'])}")

        out: List[Dict[str, torch.Tensor]] = []
        out.extend(work[:merge_idx])
        out.append(merged)
        out.extend(work[merge_idx + 2:])
        assert len(out) == max_slots
        return out

    def _choose_merge_index(self, entries: List[Dict[str, torch.Tensor]], current_frame: int) -> int:
        """Pick the adjacent pair with the smallest warped-age gap.

        Both endpoints participate in the schedule:
        * Anchor (frame 0) competes through ``gaps[0]``; identity is preserved
          inside the anchor short-circuit whenever it is selected.
        * The just-inserted slot gets one step of grace via ``protect_newest``,
          which simply excludes the rightmost gap.
        """
        assert len(entries) >= 2
        ages = [max(0.0, float(current_frame - e["center_frame"])) for e in entries]
        warped = [self._warp_age_power(a) for a in ages]
        gaps = [abs(warped[i] - warped[i + 1]) for i in range(len(warped) - 1)]
        upper = len(gaps) - 1 if (self.cfg.protect_newest and len(gaps) > 1) else len(gaps)
        return int(min(range(upper), key=lambda i: gaps[i]))

    def _is_anchor_entry(self, entry: Dict[str, torch.Tensor]) -> bool:
        """Anchor detector. Convention: only frame 0 ever has ``kv_frame == 0``,
        and the anchor short-circuit in ``_insert_entry`` carries that identity
        forward, so the test is sufficient as long as ``anchor_slots > 0``.
        """
        if int(self.cfg.anchor_slots) <= 0:
            return False
        kv = int(entry.get("kv_frame", entry.get("center_frame", -1)))
        return kv == 0

    def _warp_age_power(self, age: float) -> float:
        """Power-law age warp: u(a) = a ** p, with 0 < p < 1.

        Concave in `a`, polynomial span growth (~ a ** (1 - p)). `warp_p` is
        the only knob in this mode.
        """
        p = float(self.cfg.warp_beta)
        a = max(0.0, float(age))
        return a ** p if a > 0.0 else 0.0

    def _effective_span(self, span: int) -> float:
        """Sub-linear effective span for a single entry."""
        gamma = float(self.cfg.span_gamma)
        return float(span) ** gamma

    def _effective_span_weights(
        self, span_a: int, span_b: int
    ) -> Tuple[float, float]:
        """Compute merge weights using non-linear effective span.

        Video is non-stationary: a larger span does NOT mean higher confidence
        in the summary, it means a wider (and therefore more blurred) temporal
        average.  Linear span weights let old summaries dominate indefinitely.

        A sub-linear power lets old summaries gain influence more slowly than
        linear span weighting.
        """
        sa = self._effective_span(span_a)
        sb = self._effective_span(span_b)
        total = sa + sb
        return sa / total, sb / total

    def _merge_two_entries(
        self,
        a: Dict[str, torch.Tensor],
        b: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        span_a = max(1, int(a["span"]))
        span_b = max(1, int(b["span"]))
        total = span_a + span_b

        raw_wa, raw_wb = span_a / total, span_b / total
        center = int(round(raw_wa * float(a["center_frame"]) + raw_wb * float(b["center_frame"])))
        wa, wb = self._effective_span_weights(span_a, span_b)

        return {
            "k": wa * a["k"] + wb * b["k"],
            "v": wa * a["v"] + wb * b["v"],
            "center_frame": center,
            "kv_frame": center,
            "span": total,
        }

    def _take_newest_that_fit(self, entries: List[Dict[str, torch.Tensor]], max_tokens: int) -> List[Dict[str, torch.Tensor]]:
        """Token-budget clipping that preserves schedule boundary semantics.

        With ``anchor_slots > 0``, first keep the anchor slots that fit, then
        fill the remaining budget from the newest entries.  Returned entries
        stay in chronological order.
        """
        if max_tokens <= 0 or len(entries) == 0:
            return []

        anchor = min(max(0, int(self.cfg.anchor_slots)), len(entries))
        kept_indices = set()
        used = 0

        for i in range(anchor):
            tokens = int(entries[i]["k"].shape[1])
            if used + tokens <= max_tokens:
                kept_indices.add(i)
                used += tokens

        for i in range(len(entries) - 1, anchor - 1, -1):
            tokens = int(entries[i]["k"].shape[1])
            if used + tokens > max_tokens:
                continue
            kept_indices.add(i)
            used += tokens

        return [entry for i, entry in enumerate(entries) if i in kept_indices]
