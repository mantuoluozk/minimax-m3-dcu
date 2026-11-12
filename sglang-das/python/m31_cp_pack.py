"""M3.1 CP packet layout using the current M3 fused writer's FP32 contract."""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _pack(k_ptr, v_ptr, ik_ptr, fp8_ptr, bf16_ptr,
          ks, vs, iks, k_scale, v_scale, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK).to(tl.int64)
    k = tl.load(k_ptr + row * ks + cols).to(tl.float32)
    v = tl.load(v_ptr + row * vs + cols).to(tl.float32)
    k = tl.maximum(-448.0, tl.minimum(448.0, k / k_scale))
    v = tl.maximum(-448.0, tl.minimum(448.0, v / v_scale))
    tl.store(fp8_ptr + row * 1280 + cols, k)
    tl.store(fp8_ptr + row * 1280 + 512 + cols, v)
    ik = tl.load(ik_ptr + row * iks + cols, mask=cols < 128, other=0)
    tl.store(bf16_ptr + row * 640 + 512 + cols, ik, mask=cols < 128)


def pack_current(k, v, ik, k_scale=None, v_scale=None):
    assert all(x.is_cuda and x.dtype == torch.bfloat16 and x.ndim == 2 for x in (k, v, ik))
    assert k.shape == v.shape and k.shape[1] == 512 and ik.shape == (k.shape[0], 128)
    assert k.device == v.device == ik.device
    assert all(x.stride(1) == 1 for x in (k, v, ik))
    ks = 1.0 if k_scale is None else float(k_scale)
    vs = 1.0 if v_scale is None else float(v_scale)
    assert all(math.isfinite(s) and 1e-37 < s < 1e37 for s in (ks, vs))
    result = torch.empty((k.shape[0], 1280), device=k.device, dtype=torch.uint8)
    if k.shape[0]:
        _pack[(k.shape[0],)](k, v, ik, result.view(torch.float8_e4m3fn),
                             result.view(torch.bfloat16), k.stride(0), v.stride(0),
                             ik.stride(0), ks, vs, BLOCK=512, num_warps=4)
    return result
