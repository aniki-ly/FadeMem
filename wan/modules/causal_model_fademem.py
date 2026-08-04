# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import torch.distributed as dist
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, log_gpu_memory

from utils.debug_option import DEBUG

try:
    from fademem_memory import (
        ContinuousFadeMem,
        FADEMEM_ROPE_CHUNK_FRAMES,
        FADEMEM_ROPE_CURRENT_START,
        FadeMemConfig,
    )
except ImportError:
    from .fademem_memory import (
        ContinuousFadeMem,
        FADEMEM_ROPE_CHUNK_FRAMES,
        FADEMEM_ROPE_CURRENT_START,
        FadeMemConfig,
    )

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


# Gradient checkpointing recomputes earlier blocks after cache indices have
# advanced. Preserve the forward indices so recomputation sees the same layout.
_checkpoint_forward_state: dict = {}


def clear_checkpoint_forward_state():
    _checkpoint_forward_state.clear()


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


def _temporal_rope_complex_dims(head_dim: int) -> int:
    c = head_dim // 2
    return c - 2 * (c // 3)


def rerotate_temporal_rope(
    k_roped: torch.Tensor,
    freqs: torch.Tensor,
    old_frame_positions: torch.Tensor,
    new_frame_positions: torch.Tensor,
    frame_seqlen: int,
) -> torch.Tensor:
    """Move only the temporal RoPE band from old frame positions to new ones."""
    if k_roped.numel() == 0:
        return k_roped
    if frame_seqlen <= 0:
        raise ValueError(f"frame_seqlen must be positive, got {frame_seqlen}")
    if k_roped.shape[1] % frame_seqlen != 0:
        raise ValueError(
            f"k_roped length {k_roped.shape[1]} is not divisible by frame_seqlen {frame_seqlen}"
        )

    frame_count = k_roped.shape[1] // frame_seqlen
    if old_frame_positions.numel() != frame_count or new_frame_positions.numel() != frame_count:
        raise ValueError(
            "old/new frame position counts must match k_roped frame count: "
            f"old={old_frame_positions.numel()} new={new_frame_positions.numel()} frames={frame_count}"
        )

    head_dim = k_roped.shape[-1]
    tc = _temporal_rope_complex_dims(head_dim)
    if tc <= 0:
        return k_roped

    old_pos = old_frame_positions.to(device=freqs.device, dtype=torch.long).clamp_(0, freqs.shape[0] - 1)
    new_pos = new_frame_positions.to(device=freqs.device, dtype=torch.long).clamp_(0, freqs.shape[0] - 1)
    k_view = k_roped.view(k_roped.shape[0], frame_count, frame_seqlen, k_roped.shape[2], head_dim)
    k_complex = torch.view_as_complex(
        k_view.to(torch.float64).contiguous().reshape(*k_view.shape[:-1], head_dim // 2, 2)
    )
    rot = (freqs[new_pos, :tc] * freqs[old_pos, :tc].conj()).to(k_complex.device)
    k_complex[..., :tc] = k_complex[..., :tc] * rot.view(1, frame_count, 1, 1, tc)
    return torch.view_as_real(k_complex).flatten(-2).reshape_as(k_roped).type_as(k_roped)


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6,
                 fademem_cfg=None):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        # Support list/tuple local_attn_size by converting to list first (handles OmegaConf ListConfig)
        if not isinstance(local_attn_size, int) and hasattr(local_attn_size, "__iter__"):
            values = list(local_attn_size)
        else:
            values = [int(local_attn_size)]
        non_neg_vals = [int(v) for v in values if int(v) != -1]
        max_local = max(non_neg_vals) if len(non_neg_vals) > 0 else -1
        self.max_attention_size = 32760 if max_local == -1 else max_local * 1560
        if fademem_cfg is None:
            fademem_cfg = FadeMemConfig(enabled=False)
        elif isinstance(fademem_cfg, dict):
            fademem_cfg = FadeMemConfig(**fademem_cfg)
        self.fademem_cfg = fademem_cfg
        if DEBUG:
            print(self.fademem_cfg)
        self.fademem = ContinuousFadeMem(self.fademem_cfg)
        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def _online_rerope_current_start(self):
        if not self.fademem_cfg.enabled:
            return None
        return FADEMEM_ROPE_CURRENT_START

    def _online_rerope_attention_ready(self, kv_cache, cache_update_info, current_start_frame=None) -> bool:
        target_start = self._online_rerope_current_start()
        if target_start is None:
            return False
        if current_start_frame is None:
            return False
        if int(current_start_frame) < int(target_start):
            return False
        if kv_cache is None or "fademem_count" not in kv_cache:
            return False
        count = int(kv_cache["fademem_count"].item())
        update = None if cache_update_info is None else cache_update_info.get("fademem_update")
        if update is not None and "center_frame" in update:
            count += int(update["center_frame"].shape[0])
        return count >= int(getattr(self.fademem_cfg, "summary_slots", 0))

    def _build_online_rerope_attention_k(
        self,
        temp_k_abs,
        roped_key_attn,
        freqs,
        current_start_frame,
        current_end,
        local_end_index,
        write_start_index,
        write_len,
        roped_offset,
        sink_tokens,
        frame_seqlen,
        target_current_start,
        num_new_tokens,
    ):
        if sink_tokens != 0:
            raise ValueError("online re-rope currently requires sink_size=0")
        if local_end_index == 0:
            return temp_k_abs
        if local_end_index % frame_seqlen != 0:
            raise ValueError(
                f"local_end_index {local_end_index} must be frame aligned for online re-rope"
            )
        if num_new_tokens % frame_seqlen != 0:
            raise ValueError(
                f"num_new_tokens {num_new_tokens} must be frame aligned for online re-rope"
            )

        local_frames = local_end_index // frame_seqlen
        new_frames = num_new_tokens // frame_seqlen
        expected_local_frames = FADEMEM_ROPE_CHUNK_FRAMES
        if local_frames != expected_local_frames:
            raise ValueError(
                f"released online re-rope requires local_attn_size == {expected_local_frames}; "
                f"got local_frames={local_frames}"
            )

        old_start_frame = (current_end - local_end_index) // frame_seqlen
        old_frames = torch.arange(
            old_start_frame,
            old_start_frame + local_frames,
            device=temp_k_abs.device,
            dtype=torch.long,
        )
        target_end = int(target_current_start) + new_frames
        target_start = target_end - local_frames
        if target_start < 0:
            raise ValueError(
                f"online re-rope target_start became negative: {target_start}"
            )
        target_frames = torch.arange(
            target_start,
            target_end,
            device=temp_k_abs.device,
            dtype=torch.long,
        )

        temp_k_attn = temp_k_abs.clone()
        temp_k_attn[:, :local_end_index] = rerotate_temporal_rope(
            temp_k_abs[:, :local_end_index],
            freqs,
            old_frames,
            target_frames,
            frame_seqlen,
        )
        if write_len > 0:
            temp_k_attn[:, write_start_index:write_start_index + write_len] = (
                roped_key_attn[:, roped_offset:roped_offset + write_len]
            )
        return temp_k_attn

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        sink_recache_after_switch=False,
        fademem_snapshot=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)
        else:
            frame_h = int(grid_sizes[0][1].item())
            frame_w = int(grid_sizes[0][2].item())
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            if self.fademem_cfg.enabled:
                self.fademem.ensure_cache(
                    kv_cache,
                    batch_size=b,
                    num_heads=n,
                    head_dim=d,
                    frame_h=frame_h,
                    frame_w=frame_w,
                    device=k.device,
                    dtype=k.dtype,
                )
            # Start from the LongLive absolute rolling view. Online re-rope is
            # enabled later only after the summary bank is full for this
            # attention call.
            roped_key_abs = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            if self.fademem_cfg.enabled:
                q_start = self.fademem.rope_chunk_base(current_start_frame)
            else:
                q_start = current_start_frame
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=q_start).type_as(v)
            roped_key = roped_key_abs

            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]
            # if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
            #     print("***********before attention***********")
            #     print(f"kv_cache_size = {kv_cache_size / frame_seqlen}")
            #     print(f"torch.is_grad_enabled() = {torch.is_grad_enabled()}")
            #     print(f"current_end = {current_end / frame_seqlen}")
            #     print(f"current_start = {current_start / frame_seqlen}")
            #     print(f"kv_cache['global_end_index'] = {kv_cache['global_end_index']}")
            #     print(f"kv_cache['local_end_index'] = {kv_cache['local_end_index']}")
            #     print(f"num_new_tokens = {num_new_tokens}")

            # Checkpoint backward recomputation fix: between forward and
            # backward, the outer training loop modifies global_end_index and
            # local_end_index for subsequent blocks.  During backward
            # recomputation of an earlier block, these mutated indices produce
            # a wrong local_end_index (and thus wrong tensor shapes), crashing
            # with CheckpointError.
            #
            # Fix: on the first (forward) call we save local_end_index.  On
            # the second (backward recompute) call we inject "fake" indices
            # into the cache so that the existing code naturally computes the
            # saved local_end_index AND skips rolling (the cache has already
            # been rolled by _apply_cache_updates).  After the computation we
            # restore the real indices.
            checkpoint_key = (id(self), current_start)
            saved_forward_state = (
                _checkpoint_forward_state.pop(checkpoint_key, None)
                if torch.is_grad_enabled()
                else None
            )
            original_cache_indices = None
            if saved_forward_state is not None:
                original_cache_indices = (
                    kv_cache["global_end_index"].item(),
                    kv_cache["local_end_index"].item(),
                )
                saved_local_end_index = saved_forward_state["local_end_index"]
                # fake global_end = current_start  →  current_end > fake  →  not is_recompute
                # fake local_end  chosen so that direct_insert gives saved local_end_index
                kv_cache["global_end_index"].fill_(current_end - num_new_tokens)
                kv_cache["local_end_index"].fill_(saved_local_end_index - num_new_tokens)

            # Compute cache update parameters without modifying kv_cache directly
            cache_update_info = None
            is_recompute = current_end <= kv_cache["global_end_index"].item() and current_start > 0
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                # if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
                #     print(f"need roll")
                #     print(f"num_rolled_tokens: {num_rolled_tokens / frame_seqlen}")
                #     print(f"num_evicted_tokens: {num_evicted_tokens / frame_seqlen}")
                #     print(f"sink_tokens: {sink_tokens / frame_seqlen}")

                # Compute updated local indices
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens

                # Construct full k, v for attention computation (without modifying the original cache)
                # Create temporary k, v for computation
                temp_k_abs = kv_cache["k"].clone()
                temp_v = kv_cache["v"].clone()
                
                # Apply rolling update to the temporary cache
                temp_k_abs[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    temp_k_abs[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                temp_v[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    temp_v[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                
                # Insert new key/value into the temporary cache
                # Protect sink_tokens only during recomputation; regular forward generation allows writing into the initial sink region
                write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
                roped_offset = max(0, write_start_index - local_start_index)
                write_len = max(0, local_end_index - write_start_index)
                if write_len > 0:
                    temp_k_abs[:, write_start_index:local_end_index] = roped_key_abs[:, roped_offset:roped_offset + write_len]
                    temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]

                fademem_update = None
                if self.fademem_cfg.enabled and (not is_recompute) and num_evicted_tokens > 0:
                    evicted_abs_start = kv_cache["global_end_index"].item() - kv_cache["local_end_index"].item() + sink_tokens
                    evicted_k = kv_cache["k"][:, sink_tokens:sink_tokens + num_evicted_tokens].clone()
                    evicted_v = kv_cache["v"][:, sink_tokens:sink_tokens + num_evicted_tokens].clone()
                    # 6.22: un-rotate base must match the base the evicted chunk was
                    # roped at. Reset modes pin every chunk to W-n; absolute modes
                    # use the true absolute frame index.
                    # chunk is always roped at its true absolute position, so the
                    # evicted chunk must be un-rotated at the same absolute base
                    # (identical to the original FadeMem).
                    unrope_start_frame = evicted_abs_start // frame_seqlen
                    raw_fademem_update = self.fademem.compress_evicted_tokens(
                        evicted_k=evicted_k,
                        evicted_v=evicted_v,
                        frame_h=frame_h,
                        frame_w=frame_w,
                        start_frame=unrope_start_frame,
                        freqs=freqs,
                    )
                    if raw_fademem_update is not None:
                        fademem_update = dict(raw_fademem_update)

                # Save cache update info for later use
                cache_update_info = {
                    "action": "roll_and_insert",
                    "sink_tokens": sink_tokens,
                    "num_rolled_tokens": num_rolled_tokens,
                    "num_evicted_tokens": num_evicted_tokens,
                    "local_start_index": local_start_index,
                    "local_end_index": local_end_index,
                    "write_start_index": write_start_index,
                    "write_end_index": local_end_index,
                    "new_k": roped_key_abs[:, roped_offset:roped_offset + write_len],
                    "new_v": v[:, roped_offset:roped_offset + write_len],
                    "current_end": current_end,
                    "current_frame": current_end // frame_seqlen,
                    "fademem_update": fademem_update,
                    "is_recompute": is_recompute
                }

                # if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
                #     print(f"used kv cache size: local_end_index - local_start_index = {local_end_index - local_start_index}")
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens

                # Construct full k, v for attention computation (without modifying the original cache)
                temp_k_abs = kv_cache["k"].clone()
                temp_v = kv_cache["v"].clone()
                # Protect sink_tokens only during recomputation; regular forward generation allows writing into the initial sink region
                write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
                if sink_recache_after_switch:
                    write_start_index = local_start_index
                roped_offset = max(0, write_start_index - local_start_index)
                write_len = max(0, local_end_index - write_start_index)
                if write_len > 0:
                    temp_k_abs[:, write_start_index:local_end_index] = roped_key_abs[:, roped_offset:roped_offset + write_len]
                    temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]

                fademem_update = None

                # Save cache update info for later use
                cache_update_info = {
                    "action": "direct_insert",
                    "local_start_index": local_start_index,
                    "local_end_index": local_end_index,
                    "write_start_index": write_start_index,
                    "write_end_index": local_end_index,
                    "new_k": roped_key_abs[:, roped_offset:roped_offset + write_len],
                    "new_v": v[:, roped_offset:roped_offset + write_len],
                    "current_end": current_end,
                    "current_frame": current_end // frame_seqlen,
                    "fademem_update": fademem_update,
                    "is_recompute": is_recompute
                }

            # if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
            #     print(f"local_start_index: {local_start_index}, local_end_index: {local_end_index}")

            # Restore real cache indices after checkpoint backward recomputation
            if original_cache_indices is not None:
                kv_cache["global_end_index"].fill_(original_cache_indices[0])
                kv_cache["local_end_index"].fill_(original_cache_indices[1])
            elif torch.is_grad_enabled():
                _checkpoint_forward_state[(id(self), current_start)] = {
                    "local_end_index": local_end_index,
                }

            summary_k_segments = None
            summary_v_segments = None
            if self.fademem_cfg.enabled:
                local_offset = self.local_attn_size if self.local_attn_size > 0 else 0
                fademem_anchor = max(0, current_end // frame_seqlen - local_offset)
                _fademem_update_for_attn = cache_update_info.get("fademem_update")
                if fademem_snapshot is not None:
                    # FIX: under checkpointing, backward recomputation cannot
                    # reproduce the eviction-based fademem_update (the evicted
                    # KV has been overwritten). Use snapshot-only entries so
                    # forward and backward see identical fademem prefix.
                    _fademem_update_for_attn = None
                summary_k_segments, summary_v_segments = self.fademem.build_attention_prefix_segments(
                    cache=kv_cache,
                    update_info=_fademem_update_for_attn,
                    current_frame=fademem_anchor,
                    max_tokens=None,
                    snapshot=fademem_snapshot,
                    freqs=freqs,
                    query_frame=current_start_frame,
                )

            online_target_start = None
            if self.fademem_cfg.enabled and self._online_rerope_attention_ready(kv_cache, cache_update_info, current_start_frame):
                online_target_start = int(self._online_rerope_current_start())
                if online_target_start != q_start:
                    roped_query = causal_rope_apply(
                        q, grid_sizes, freqs, start_frame=online_target_start).type_as(v)
                    roped_key = causal_rope_apply(
                        k, grid_sizes, freqs, start_frame=online_target_start).type_as(v)

            if online_target_start is not None:
                temp_k_attn = self._build_online_rerope_attention_k(
                    temp_k_abs=temp_k_abs,
                    roped_key_attn=roped_key,
                    freqs=freqs,
                    current_start_frame=current_start_frame,
                    current_end=current_end,
                    local_end_index=local_end_index,
                    write_start_index=write_start_index,
                    write_len=write_len,
                    roped_offset=roped_offset,
                    sink_tokens=sink_tokens,
                    frame_seqlen=frame_seqlen,
                    target_current_start=int(online_target_start),
                    num_new_tokens=num_new_tokens,
                )
            else:
                temp_k_attn = temp_k_abs

            if self.fademem_cfg.enabled:
                # No budget competition: concatenate sink + ALL fademem + ALL local
                prefix_k = []
                prefix_v = []
                if sink_tokens > 0:
                    prefix_k.append(temp_k_attn[:, :sink_tokens])
                    prefix_v.append(temp_v[:, :sink_tokens])
                if summary_k_segments is not None:
                    prefix_k.extend(summary_k_segments)
                    prefix_v.extend(summary_v_segments)
                if local_end_index > sink_tokens:
                    prefix_k.append(temp_k_attn[:, sink_tokens:local_end_index])
                    prefix_v.append(temp_v[:, sink_tokens:local_end_index])
                k_cat = torch.cat(prefix_k, dim=1)
                v_cat = torch.cat(prefix_v, dim=1)
                x = attention(roped_query, k_cat, v_cat)
            elif sink_tokens > 0:
                local_budget = max(0, self.max_attention_size - sink_tokens)
                prefix_k = [temp_k_attn[:, :sink_tokens]]
                prefix_v = [temp_v[:, :sink_tokens]]
                if local_budget > 0:
                    local_start_for_window = max(sink_tokens, local_end_index - local_budget)
                    prefix_k.append(temp_k_attn[:, local_start_for_window:local_end_index])
                    prefix_v.append(temp_v[:, local_start_for_window:local_end_index])
                k_cat = torch.cat(prefix_k, dim=1)
                v_cat = torch.cat(prefix_v, dim=1)
                x = attention(roped_query, k_cat, v_cat)
            else:
                local_budget = self.max_attention_size
                local_start = max(0, local_end_index - local_budget)
                x = attention(
                    roped_query,
                    temp_k_attn[:, local_start:local_end_index],
                    temp_v[:, local_start:local_end_index]
                )
        # output
        x = x.flatten(2)
        x = self.o(x)
        
        # Return both output and cache update info
        if kv_cache is not None:
            return x, (current_end, local_end_index, cache_update_info)
        else:
            return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 fademem_cfg=None):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps, fademem_cfg=fademem_cfg)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        sink_recache_after_switch=False,
        fademem_snapshot=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        self_attn_result = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start, sink_recache_after_switch, fademem_snapshot)
        
        if kv_cache is not None:
            y, cache_update_info = self_attn_result
        else:
            y = self_attn_result
            cache_update_info = None

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        
        if cache_update_info is not None:
            # cache_update_info is already in the format (current_end, local_end_index, cache_update_info)
            return x, cache_update_info
        else:
            return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 fademem_enabled=False,
                 fademem_summary_slots=8,
                 fademem_warp_type='power',
                 fademem_warp_beta=1.0,
                 fademem_span_gamma=0.5,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.fademem_cfg = FadeMemConfig(
            enabled=bool(fademem_enabled),
            summary_slots=int(fademem_summary_slots),
            warp_type=str(fademem_warp_type).lower(),
            warp_beta=float(fademem_warp_beta),
            span_gamma=float(fademem_span_gamma),
        )

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps,
                                    fademem_cfg=self.fademem_cfg)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _ensure_freqs_length(self, min_len: int, device) -> None:
        """Extend deterministic RoPE tables when absolute cache storage exceeds 1024."""
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        if min_len <= self.freqs.shape[0]:
            return
        d = self.dim // self.num_heads
        new_len = max(int(min_len), int(self.freqs.shape[0]) * 2)
        self.freqs = torch.cat([
            rope_params(new_len, d - 4 * (d // 6)),
            rope_params(new_len, 2 * (d // 6)),
            rope_params(new_len, 2 * (d // 6))
        ], dim=1).to(device)

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
            pass

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        # # debug
        # DEBUG = False
        # if DEBUG:
        #     num_frames = 9
        #     frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                               padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            pass

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    def _apply_cache_updates(self, kv_cache, cache_update_infos):
        """
        Applies cache updates collected from multiple blocks.
        Args:
            kv_cache: List of cache dictionaries for each block
            cache_update_infos: List of (block_index, cache_update_info) tuples
        """

        def _detach_update_tree(obj):
            if torch.is_tensor(obj):
                return obj.detach()
            if isinstance(obj, dict):
                return {k: _detach_update_tree(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                t = [_detach_update_tree(v) for v in obj]
                return type(obj)(t) if isinstance(obj, tuple) else t
            return obj

        with torch.no_grad():
            for block_index, (current_end, local_end_index, update_info) in cache_update_infos:
                if update_info is not None:
                    cache = kv_cache[block_index]
                    safe_update = _detach_update_tree(update_info)

                    if safe_update["action"] == "roll_and_insert":
                        sink_tokens = safe_update["sink_tokens"]
                        num_rolled_tokens = safe_update["num_rolled_tokens"]
                        num_evicted_tokens = safe_update["num_evicted_tokens"]
                        local_start_index = safe_update["local_start_index"]
                        local_end_index = safe_update["local_end_index"]
                        write_start_index = safe_update.get("write_start_index", local_start_index)
                        write_end_index = safe_update.get("write_end_index", local_end_index)
                        new_k = safe_update["new_k"]
                        new_v = safe_update["new_v"]

                        cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                            cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                        cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                            cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()

                        if write_end_index > write_start_index and new_k.shape[1] == (write_end_index - write_start_index):
                            cache["k"][:, write_start_index:write_end_index] = new_k
                            cache["v"][:, write_start_index:write_end_index] = new_v

                    elif safe_update["action"] == "direct_insert":
                        local_start_index = safe_update["local_start_index"]
                        local_end_index = safe_update["local_end_index"]
                        write_start_index = safe_update.get("write_start_index", local_start_index)
                        write_end_index = safe_update.get("write_end_index", local_end_index)
                        new_k = safe_update["new_k"]
                        new_v = safe_update["new_v"]

                        if write_end_index > write_start_index and new_k.shape[1] == (write_end_index - write_start_index):
                            cache["k"][:, write_start_index:write_end_index] = new_k
                            cache["v"][:, write_start_index:write_end_index] = new_v

                    fademem_update = safe_update.get("fademem_update")
                    if fademem_update is not None and getattr(self.blocks[block_index].self_attn.fademem_cfg, "enabled", False):
                        raw_frame = int(safe_update.get("current_frame", current_end))
                        local_attn = self.blocks[block_index].self_attn.local_attn_size
                        fademem_anchor = max(0, raw_frame - (local_attn if local_attn > 0 else 0))
                        fademem_module = self.blocks[block_index].self_attn.fademem
                        fademem_module.apply_update(
                            cache=cache,
                            update_info=fademem_update,
                            current_frame=fademem_anchor,
                        )

                is_recompute = False if update_info is None else update_info.get("is_recompute", False)
                if not is_recompute:
                    kv_cache[block_index]["global_end_index"].fill_(current_end)
                    kv_cache[block_index]["local_end_index"].fill_(local_end_index)

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        sink_recache_after_switch=False
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        
        # print(f"x.device: {x[0].device}, t.device: {t.device}, context.device: {context.device}, seq_len: {seq_len}")

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        # print("patch embedding done")
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        if current_start:
            frame_seqlen_for_freqs = int(math.prod(grid_sizes[0][1:]).item())
            current_start_frame_for_freqs = int(current_start // frame_seqlen_for_freqs)
        else:
            current_start_frame_for_freqs = 0
        required_freq_len = current_start_frame_for_freqs + int(grid_sizes[:, 0].max().item())
        self._ensure_freqs_length(required_freq_len, device)
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32
        # print("time embedding done")
        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))
        # print("text embedding done")
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            sink_recache_after_switch=sink_recache_after_switch
        )
        # print("kwargs done")

        fademem_snapshots = None
        if (
            kv_cache is not None
            and torch.is_grad_enabled()
            and self.gradient_checkpointing
            and getattr(self.fademem_cfg, "enabled", False)
        ):
            fademem_snapshots = []
            for block_index in range(len(self.blocks)):
                cache_block = kv_cache[block_index]
                fademem_module = self.blocks[block_index].self_attn.fademem
                fademem_snapshots.append(fademem_module.clone_snapshot(cache_block))

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        cache_update_info = None
        cache_update_infos = []  # Collect cache update info for all blocks
        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "fademem_snapshot": None if fademem_snapshots is None else fademem_snapshots[block_index],
                    }
                )
                result = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
                if kv_cache is not None and isinstance(result, tuple):
                    x, block_cache_update_info = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                    cache_update_info = block_cache_update_info[:2]
                else:
                    x = result
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "fademem_snapshot": None,
                    }
                )
                result = block(x, **kwargs)
                if kv_cache is not None and isinstance(result, tuple):
                    x, block_cache_update_info = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                    cache_update_info = block_cache_update_info[:2]
                else:
                    x = result
        # log_gpu_memory(f"in _forward_inference: {x[0].device}")
        # After all blocks are processed, apply cache updates in a single pass
        if kv_cache is not None and cache_update_infos:
            self._apply_cache_updates(kv_cache, cache_update_infos)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        pass
        raise NotImplementedError()
    
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)
        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
