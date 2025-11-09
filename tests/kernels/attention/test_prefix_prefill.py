# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import random
import time
from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F
from vllm.attention.ops.chunked_prefill_paged_decode import chunked_prefill_paged_decode
from vllm.attention.ops.prefix_prefill import context_attention_fwd
from vllm.platforms import current_platform
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

NUM_HEADS = [64]
NUM_QUERIES_PER_KV = [1, 64]
HEAD_SIZES = [24, 128]
DTYPES = [torch.float16]
CUDA_DEVICES = [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]
SLIDING_WINDOW = [0, 16, 2048]
KV_CACHE_DTYPES = ["auto", "fp8", "fp8_e5m2"]

OPS = [chunked_prefill_paged_decode, context_attention_fwd]


def _build_sdpa_mask(
    num_heads: int,
    ctx_len: int,
    query_len: int,
    seq_len: int,
    sliding_window: int,
    device: torch.device,
    alibi_slopes: torch.Tensor | None,
) -> torch.Tensor:
    """Construct an attention mask that matches vLLM's prefix prefill rules.

    The Triton kernels only allow each query token to attend to:

    * The *entire* context (``ctx_len`` tokens) that has already been
      prefetched into KV cache.
    * Its own prefix tokens when we are doing sliding window attention.

    In PyTorch's SDPA API the mask is additive, so we fill the disallowed
    locations with ``-inf`` and the allowed ones with ``0``.
    """
    mask = torch.full(
        (query_len, seq_len), float("-inf"), device=device, dtype=torch.float32
    )
    for query_idx in range(query_len):
        # Every new query token can look at all prior context plus the tokens
        # generated within the sliding window immediately before it.
        allowed_end = ctx_len + query_idx + 1
        if sliding_window > 0:
            allowed_start = max(0, allowed_end - sliding_window)
        else:
            allowed_start = 0
        allowed_start = min(allowed_start, allowed_end)
        mask[query_idx, allowed_start:allowed_end] = 0.0

    # Expand to ``(num_heads, query_len, seq_len)`` so SDPA can broadcast the
    # mask per attention head, then add an explicit batch dimension.
    attn_mask = mask.unsqueeze(0).expand(num_heads, -1, -1).clone()

    if alibi_slopes is not None:
        # When ALiBi is enabled the mask carries both the additive ``-inf``
        # entries *and* the positional bias.  SDPA expects the bias to be of
        # shape ``(num_heads, query_len, seq_len)``, so we compute it here and
        # add it into the mask tensor we already created.
        pos_q = torch.arange(
            ctx_len,
            ctx_len + query_len,
            device=device,
            dtype=torch.float32,
        )
        pos_k = torch.arange(seq_len, device=device, dtype=torch.float32)
        bias = pos_q[:, None] - pos_k[None, :]
        attn_mask += alibi_slopes.to(torch.float32)[:, None, None] * bias

    return attn_mask.unsqueeze(0)


def _sdpa_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_lens: list[int],
    ctx_lens: list[int],
    seq_lens: list[int],
    num_heads: int,
    num_kv_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    sliding_window: int,
    alibi_slopes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a SDPA forward pass that mirrors the Triton prefix prefill kernel.

    The prefix prefill kernel processes a flattened batch of requests where the
    context tokens are already in KV cache.  To produce a numerically comparable
    result we replicate the same data layout manipulations here before calling
    ``torch.nn.functional.scaled_dot_product_attention``.
    """
    output_ref = torch.empty_like(query)

    query_offset = 0
    key_offset = 0

    for ctx_len, query_len, seq_len in zip(ctx_lens, query_lens, seq_lens):
        # Slice out a single request from the flattened batch.  The Triton
        # kernels operate on flattened tensors, so we mimic that layout rather
        # than reshaping into ``(batch, time, head, dim)`` as a user-facing
        # model would do.
        q_slice = query[query_offset:query_offset + query_len]
        k_slice = key[key_offset:key_offset + seq_len]
        v_slice = value[key_offset:key_offset + seq_len]

        if num_kv_heads != num_heads:
            # Multi-query attention stores fewer KV heads than attention heads.
            # The CUDA kernels internally broadcast the cached KV heads across
            # all query heads, so we reproduce the exact expansion here to keep
            # the SDPA reference aligned with the Triton implementation.
            k_slice = k_slice[:, :, None, :].expand(
                seq_len, num_kv_heads, num_queries_per_kv, head_size
            )
            k_slice = k_slice.reshape(seq_len, num_heads, head_size).contiguous()
            v_slice = v_slice[:, :, None, :].expand(
                seq_len, num_kv_heads, num_queries_per_kv, head_size
            )
            v_slice = v_slice.reshape(seq_len, num_heads, head_size).contiguous()

        # SDPA expects the shape ``(batch, num_heads, time, head_size)``.  We
        # treat each request as a batch of size one to match the Triton kernel
        # semantics where batches do not interact.
        q_formatted = q_slice.permute(1, 0, 2).unsqueeze(0)
        k_formatted = k_slice.permute(1, 0, 2).unsqueeze(0)
        v_formatted = v_slice.permute(1, 0, 2).unsqueeze(0)

        attn_mask = _build_sdpa_mask(
            num_heads,
            ctx_len,
            query_len,
            seq_len,
            sliding_window,
            q_slice.device,
            alibi_slopes,
        )

        attn_output = F.scaled_dot_product_attention(
            q_formatted,
            k_formatted,
            v_formatted,
            attn_mask=attn_mask,
        )
        attn_output = attn_output.squeeze(0).permute(1, 0, 2).contiguous()

        output_ref[query_offset:query_offset + query_len] = attn_output

        query_offset += query_len
        key_offset += seq_len

    return output_ref


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOW)
@pytest.mark.parametrize("op", OPS)
@torch.inference_mode()
def test_contexted_kv_attention(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    sliding_window: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
    op: Callable,
) -> None:
    if "fp8" in kv_cache_dtype and not current_platform.has_device_capability(89):
        pytest.skip(
            "Triton limitation: fp8e4nv data type is not supported on CUDA arch < 89"
        )

    current_platform.seed_everything(0)
    torch.set_default_device(device)

    # Need this, otherwise when we capture the graph the process
    # for GPU 1 would run on both GPU0 and GPU1 and things would hang
    #
    # see also similar issue: https://github.com/Dao-AILab/flash-attention/issues/523
    torch.cuda.set_device(device)

    MAX_SEQ_LEN = 1024
    MAX_CTX_LEN = 1024
    BS = 10
    cache_size = 640
    block_size = 32
    max_block_per_request = 64
    query_lens = [random.randint(16, MAX_SEQ_LEN) for _ in range(BS)]
    # ensure one sequence in batch is a decode
    query_lens[-1] = 1

    ctx_lens = [random.randint(16, MAX_CTX_LEN) for _ in range(BS)]
    seq_lens = [a + b for a, b in zip(query_lens, ctx_lens)]
    num_kv_heads = num_heads // num_queries_per_kv

    num_tokens = sum(query_lens)
    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
    query.uniform_(-1e-3, 1e-3)
    output = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)

    kv = torch.empty(sum(seq_lens), 2, num_kv_heads, head_size, dtype=dtype)
    kv.uniform_(-1e-3, 1e-3)
    key, value = kv.unbind(dim=1)

    if kv_cache_dtype == "auto":
        cache_dtype = dtype
    else:
        cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]
    k_cache = torch.zeros(
        cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype
    )
    v_cache = torch.zeros(
        cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype
    )
    k = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    v = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    values = torch.arange(0, cache_size, dtype=torch.long)
    values = values[torch.randperm(cache_size)]
    block_table = values[: BS * max_block_per_request].view(BS, max_block_per_request)
    b_seq_len = torch.tensor(seq_lens, dtype=torch.long)
    b_ctx_len = torch.tensor(ctx_lens, dtype=torch.long)
    b_start_loc = torch.cumsum(torch.tensor([0] + query_lens, dtype=torch.long), dim=0)
    max_input_len = MAX_SEQ_LEN
    # copy kv to cache
    b_seq_start_loc = torch.cumsum(
        torch.tensor([0] + seq_lens[:-1], dtype=torch.long), dim=0
    )
    for i in range(BS):
        for j in range(query_lens[i]):
            k[b_start_loc[i] + j].copy_(key[b_seq_start_loc[i] + b_ctx_len[i] + j])
            v[b_start_loc[i] + j].copy_(value[b_seq_start_loc[i] + b_ctx_len[i] + j])
        cur_ctx = 0
        block_id = 0
        while cur_ctx < b_ctx_len[i]:
            start_loc = b_seq_start_loc[i] + cur_ctx
            if cur_ctx + block_size > b_ctx_len[i]:
                end_loc = b_seq_start_loc[i] + b_ctx_len[i]
            else:
                end_loc = start_loc + block_size
            start_slot = block_table[i, block_id] * block_size
            end_slot = start_slot + end_loc - start_loc
            k_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                key[start_loc:end_loc]
            )
            v_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                value[start_loc:end_loc]
            )
            cur_ctx += block_size
            block_id += 1
    # transpose K_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to K_cache[num_blocks, num_kv_heads, head_size/8, block_size, 8]
    k_cache = (
        k_cache.view(-1, block_size, num_kv_heads, head_size // 8, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    # transpose V_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to V_cache[num_blocks, num_kv_heads, head_size, block_size]
    v_cache = (
        v_cache.view(-1, block_size, num_kv_heads, head_size)
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Warm up the Triton kernel by calling it once before actually measuring
    # generation time
    op(
        query,
        k,
        v,
        output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        MAX_CTX_LEN,
        max_input_len,
        k_scale,
        v_scale,
        sliding_window=sliding_window,
    )
    torch.cuda.synchronize()
    start_time = time.time()
    op(
        query,
        k,
        v,
        output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        MAX_CTX_LEN,
        max_input_len,
        k_scale,
        v_scale,
        sliding_window=sliding_window,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"triton Time: {(end_time - start_time) * 1000:.2f} ms")

    torch.cuda.synchronize()
    start_time = time.time()
    output_ref = _sdpa_reference(
        query,
        key,
        value,
        query_lens,
        ctx_lens,
        seq_lens,
        num_heads,
        num_kv_heads,
        num_queries_per_kv,
        head_size,
        sliding_window,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"sdpa Time: {(end_time - start_time) * 1000:.2f} ms")
    atol = 1e-3 if "fp8" in kv_cache_dtype else 1e-4
    torch.testing.assert_close(output, output_ref, atol=atol, rtol=0)


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("op", OPS)
@torch.inference_mode()
def test_contexted_kv_attention_alibi(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
    op: Callable,
) -> None:
    if "fp8" in kv_cache_dtype and not current_platform.has_device_capability(89):
        pytest.skip(
            "Triton limitation: fp8e4nv data type is not supported on CUDA arch < 89"
        )

    current_platform.seed_everything(0)
    torch.set_default_device(device)

    # Need this, otherwise when we capture the graph the process
    # for GPU 1 would run on both GPU0 and GPU1 and things would hang
    #
    # see also similar issue: https://github.com/Dao-AILab/flash-attention/issues/523
    torch.cuda.set_device(device)

    def _get_alibi_slopes(total_num_heads: int) -> torch.Tensor:
        """Reproduce the ALiBi slope schedule used by BLOOM-style models."""
        # Fork from: vllm/vllm/model_executor/models/bloom.py#L44
        closest_power_of_2 = 2 ** math.floor(math.log2(total_num_heads))
        base = torch.tensor(
            2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
            dtype=torch.float32,
        )
        powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
        slopes = torch.pow(base, powers)

        if closest_power_of_2 != total_num_heads:
            extra_base = torch.tensor(
                2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
                dtype=torch.float32,
            )
            num_remaining_heads = min(
                closest_power_of_2, total_num_heads - closest_power_of_2
            )
            extra_powers = torch.arange(
                start=1, end=1 + 2 * num_remaining_heads, step=2, dtype=torch.int32
            )
            slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
        return slopes

    alibi_slopes = _get_alibi_slopes(num_heads).to(device)

    MAX_SEQ_LEN = 1024
    MAX_CTX_LEN = 1024
    BS = 10
    cache_size = 640
    block_size = 32
    max_block_per_request = 64
    query_lens = [random.randint(16, MAX_SEQ_LEN) for _ in range(BS)]
    ctx_lens = [random.randint(16, MAX_CTX_LEN) for _ in range(BS)]
    seq_lens = [a + b for a, b in zip(query_lens, ctx_lens)]
    num_kv_heads = num_heads // num_queries_per_kv

    num_tokens = sum(query_lens)
    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
    query.uniform_(-1e-3, 1e-3)
    output = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)

    kv = torch.empty(sum(seq_lens), 2, num_kv_heads, head_size, dtype=dtype)
    kv.uniform_(-1e-3, 1e-3)
    key, value = kv.unbind(dim=1)
    if kv_cache_dtype == "auto":
        cache_dtype = dtype
    else:
        cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]
    k_cache = torch.zeros(
        cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype
    )
    v_cache = torch.zeros(
        cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype
    )
    k = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    v = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    values = torch.arange(0, cache_size, dtype=torch.long)
    values = values[torch.randperm(cache_size)]
    block_table = values[: BS * max_block_per_request].view(BS, max_block_per_request)
    b_seq_len = torch.tensor(seq_lens, dtype=torch.long)
    b_ctx_len = torch.tensor(ctx_lens, dtype=torch.long)
    b_start_loc = torch.cumsum(torch.tensor([0] + query_lens, dtype=torch.long), dim=0)
    max_input_len = MAX_SEQ_LEN
    # Copy the first ``ctx_len`` tokens from every request into the paged KV
    # cache.  This mirrors the behavior in production inference where the
    # prefix context is resident in cache before we run the prefix prefill
    # kernel.
    b_seq_start_loc = torch.cumsum(
        torch.tensor([0] + seq_lens[:-1], dtype=torch.long), dim=0
    )
    for i in range(BS):
        for j in range(query_lens[i]):
            k[b_start_loc[i] + j].copy_(key[b_seq_start_loc[i] + b_ctx_len[i] + j])
            v[b_start_loc[i] + j].copy_(value[b_seq_start_loc[i] + b_ctx_len[i] + j])
        cur_ctx = 0
        block_id = 0
        while cur_ctx < b_ctx_len[i]:
            start_loc = b_seq_start_loc[i] + cur_ctx
            if cur_ctx + block_size > b_ctx_len[i]:
                end_loc = b_seq_start_loc[i] + b_ctx_len[i]
            else:
                end_loc = start_loc + block_size
            start_slot = block_table[i, block_id] * block_size
            end_slot = start_slot + end_loc - start_loc
            k_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                key[start_loc:end_loc]
            )
            v_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                value[start_loc:end_loc]
            )
            cur_ctx += block_size
            block_id += 1
    # Transpose the dense caches into the memory layout expected by the
    # Triton kernels.  The reshapes match the exact transformations inside the
    # implementation so the SDPA reference is compared against the same inputs.
    k_cache = (
        k_cache.view(-1, block_size, num_kv_heads, head_size // 8, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    # transpose V_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to V_cache[num_blocks, num_kv_heads, head_size, block_size]
    v_cache = (
        v_cache.view(-1, block_size, num_kv_heads, head_size)
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Warm up the Triton kernel by calling it once before actually measuring
    # generation time
    op(
        query,
        k,
        v,
        output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        MAX_CTX_LEN,
        max_input_len,
        k_scale,
        v_scale,
        alibi_slopes=alibi_slopes,
    )
    torch.cuda.synchronize()
    start_time = time.time()
    op(
        query,
        k,
        v,
        output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        MAX_CTX_LEN,
        max_input_len,
        k_scale,
        v_scale,
        alibi_slopes=alibi_slopes,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"triton Time: {(end_time - start_time) * 1000:.2f} ms")
    torch.cuda.synchronize()
    start_time = time.time()
    output_ref = _sdpa_reference(
        query,
        key,
        value,
        query_lens,
        ctx_lens,
        seq_lens,
        num_heads,
        num_kv_heads,
        num_queries_per_kv,
        head_size,
        sliding_window=0,
        alibi_slopes=alibi_slopes,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"sdpa Time: {(end_time - start_time) * 1000:.2f} ms")
    atol = 1e-3 if "fp8" in kv_cache_dtype else 1e-6
    torch.testing.assert_close(output, output_ref, atol=atol, rtol=0)


# These tests are optional to only run when explicitly invoked
#
# pytest -v -s --optional \
# tests/kernels/test_prefix_prefill.py::test_contexted_kv_attention_f32
#
# These tests are useful to test model dtype float32 on Turing devices.
# We skip them to not increase the time when running tests on CI
@pytest.mark.optional
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOW)
@pytest.mark.parametrize("op", OPS)
@torch.inference_mode()
def test_contexted_kv_attention_f32(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    sliding_window: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
    op: Callable,
) -> None:
    test_contexted_kv_attention(
        num_heads,
        num_queries_per_kv,
        head_size,
        sliding_window,
        dtype,
        kv_cache_dtype,
        device,
        op,
    )


@pytest.mark.optional
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("op", OPS)
@torch.inference_mode()
def test_contexted_kv_attention_alibi_f32(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
    op: Callable,
) -> None:
    test_contexted_kv_attention_alibi(
        num_heads, num_queries_per_kv, head_size, dtype, kv_cache_dtype, device, op
    )
