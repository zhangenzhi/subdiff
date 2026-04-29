"""Probe which SDPA kernel actually runs on H100 with our shapes/dtype.

Tests:
  1. Default SDPA (whatever PyTorch picks)
  2. Force FLASH backend
  3. Force CUDNN backend (PyTorch 2.4+; routes to cuDNN FA on Hopper)

Captures kernel names via torch.profiler. FA3 kernels in PyTorch 2.4+ live
under 'flash_fwd' / 'mha_fwd_kernel' with v3-specific tags on Hopper.
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import profile, ProfilerActivity


def time_and_name(label, q, k, v, backend=None, n_warm=3, n_iter=10):
    # Warmup
    for _ in range(n_warm):
        if backend is not None:
            with sdpa_kernel(backend):
                out = F.scaled_dot_product_attention(q, k, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()

    # Profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n_iter):
            if backend is not None:
                with sdpa_kernel(backend):
                    out = F.scaled_dot_product_attention(q, k, v)
            else:
                out = F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()

    # Pull top CUDA kernels
    print(f'\n=== {label} ===')
    # PyTorch 2.10 renamed cuda_time_total → device_time_total in some paths
    try:
        table = prof.key_averages().table(
            sort_by="device_time_total", row_limit=8)
    except Exception:
        table = prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=8)
    print(table)

    # Also dig out specific names that look attention-related
    print(f'  matching kernels:')
    for evt in prof.key_averages():
        n = evt.key
        if not any(s in n.lower() for s in
                   ['flash', 'fmha', 'mha', 'attention', 'sdp', 'cudnn']):
            continue
        # API changed across torch versions: try several attrs
        t_us = (getattr(evt, 'self_device_time_total', None)
                or getattr(evt, 'device_time_total', None)
                or getattr(evt, 'self_cuda_time_total', None)
                or getattr(evt, 'cuda_time_total', 0))
        print(f'    {t_us/1000:7.1f}us  {n}')


def main():
    print(f'PyTorch {torch.__version__}, '
          f'GPU={torch.cuda.get_device_name(0)} '
          f'(sm_{torch.cuda.get_device_capability(0)[0]}'
          f'{torch.cuda.get_device_capability(0)[1]})')

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    # Our actual shapes: ViT-B p8 224×224 → N=784 tokens, 12 heads, D=64
    B, H, N, D = 64, 12, 784, 64
    dtype = torch.bfloat16
    device = 'cuda'

    q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # 1. Default — let PyTorch pick
    time_and_name('DEFAULT (whatever PyTorch picks)', q, k, v, backend=None)

    # 2. Force FLASH
    try:
        time_and_name('FLASH_ATTENTION (FA2 / FA3 depending on torch+arch)',
                      q, k, v, backend=SDPBackend.FLASH_ATTENTION)
    except Exception as e:
        print(f'\n=== FLASH_ATTENTION: FAIL — {e}')

    # 3. Force CUDNN (PyTorch 2.4+; on Hopper routes to cuDNN FA, often FA3)
    try:
        time_and_name('CUDNN_ATTENTION', q, k, v,
                      backend=SDPBackend.CUDNN_ATTENTION)
    except Exception as e:
        print(f'\n=== CUDNN_ATTENTION: FAIL — {e}')

    # Also probe `_check_*` helpers to see what PyTorch *thinks* is usable
    from torch.nn.attention import SDPAParams
    print('\n=== SDPA backend availability check ===')
    try:
        from torch.backends.cuda import (can_use_flash_attention,
                                          can_use_efficient_attention,
                                          can_use_cudnn_attention)
        params = SDPAParams(q, k, v, None, 0.0, False, False)
        print(f'  can_use_flash_attention:     {can_use_flash_attention(params)}')
        print(f'  can_use_efficient_attention: {can_use_efficient_attention(params)}')
        print(f'  can_use_cudnn_attention:     {can_use_cudnn_attention(params)}')
    except Exception as e:
        print(f'  helper lookup failed: {e}')


if __name__ == '__main__':
    main()
