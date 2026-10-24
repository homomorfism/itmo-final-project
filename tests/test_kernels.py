import pytest
import torch

from int4_triton import (
    Int4Linear,
    dequantize_int4_ref,
    quantize_int4,
    quantize_int4_ref,
    w4a16_matmul,
    w16a16_matmul,
)
from int4_triton.quantize import unpack_int4_ref

DEV = "cuda"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(512, 2048), (2048, 8192), (100, 256)])
@pytest.mark.parametrize("group_size", [32, 128, -1])
def test_quantize_matches_reference(shape, group_size, dtype):
    torch.manual_seed(0)
    w = (torch.randn(shape, device=DEV) * 0.02).to(dtype)
    qw, s, z = quantize_int4(w, group_size)
    qw_r, s_r, z_r = quantize_int4_ref(w, group_size)
    torch.testing.assert_close(s, s_r, rtol=0, atol=0)
    # Triton's fp32 division is approximate (div.full), so values exactly on a .5 rounding
    # boundary may round differently: allow rare off-by-one differences
    dz = (z - z_r).abs()
    assert dz.max() <= 1 and (dz > 0).float().mean() < 1e-3
    diff = (unpack_int4_ref(qw) - unpack_int4_ref(qw_r)).abs()
    assert diff.max() <= 1 and (diff > 0).float().mean() < 1e-3


def test_memory_reduction_and_error():
    torch.manual_seed(0)
    N, K = 2048, 8192
    w = torch.randn(N, K, device=DEV, dtype=torch.float16) * 0.02
    qw, s, z = quantize_int4(w, 128)
    assert qw.dtype == torch.int32 and qw.shape == (K // 8, N)
    assert w.numel() * w.element_size() == 4 * qw.numel() * qw.element_size()
    w_hat = dequantize_int4_ref(qw, s, z)
    # max error is half a quantization step
    step = s.t().repeat_interleave(128, dim=1).float()
    assert ((w_hat.float() - w.float()).abs() <= step * 0.5 + 1e-3).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("M", [1, 7, 128, 512, 2048])
@pytest.mark.parametrize("NK", [(2048, 2048), (512, 2048), (8192, 2048), (2048, 8192), (300, 256)])
def test_w4a16_matmul(M, NK, dtype):
    torch.manual_seed(0)
    N, K = NK
    x = torch.randn(M, K, device=DEV, dtype=dtype)
    w = torch.randn(N, K, device=DEV, dtype=torch.float16) * 0.02
    qw, s, z = quantize_int4(w, 128)
    w_hat = dequantize_int4_ref(qw, s, z, torch.float32)
    ref = x.float() @ w_hat.t()
    out = w4a16_matmul(x, qw, s, z)
    assert out.dtype == dtype and out.shape == (M, N)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2 * ref.abs().max().item())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("M", [128, 2048])
def test_w16a16_matmul(M, dtype):
    torch.manual_seed(0)
    N, K = 2048, 8192
    x = torch.randn(M, K, device=DEV, dtype=dtype)
    w = torch.randn(N, K, device=DEV, dtype=dtype) * 0.02
    ref = x.float() @ w.float().t()
    out = w16a16_matmul(x, w)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2 * ref.abs().max().item())


def test_int4_linear_3d_input():
    torch.manual_seed(0)
    lin = torch.nn.Linear(2048, 512, bias=True, device=DEV, dtype=torch.bfloat16)
    q = Int4Linear.from_linear(lin, 128)
    x = torch.randn(2, 33, 2048, device=DEV, dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x.float(), q.dequantize(torch.float32), lin.bias.float())
    out = q(x)
    assert out.shape == (2, 33, 512)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2 * ref.abs().max().item())


@pytest.mark.parametrize("M", [1, 16, 64, 65])
@pytest.mark.parametrize("group_size", [32, 64])
def test_small_groups_strided_input_and_repeatability(M, group_size):
    torch.manual_seed(1)
    x = torch.randn(M, 512, device=DEV, dtype=torch.bfloat16)[:, ::2]
    w = torch.randn(73, 256, device=DEV, dtype=torch.bfloat16) * 0.02
    qw, s, z = quantize_int4(w, group_size)
    ref = x.float() @ dequantize_int4_ref(qw, s, z, torch.bfloat16).float().t()
    out = w4a16_matmul(x, qw, s, z)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=5e-3)
    # Reusing a compiled specialization must not depend on old scratch-buffer contents.
    torch.testing.assert_close(w4a16_matmul(x, qw, s, z), out, rtol=0, atol=0)


@pytest.mark.parametrize("M", [1, 16, 65])
def test_native_bf16_preserves_large_activation_range(M):
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("native bf16 tensor cores require Ampere or newer")
    x = torch.full((M, 128), 131072.0, device=DEV, dtype=torch.bfloat16)
    w = torch.full((73, 128), 0.5, device=DEV, dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x, w)
    torch.testing.assert_close(w16a16_matmul(x, w), ref, rtol=0, atol=0)
    qw, s, z = quantize_int4(w)
    ref4 = torch.nn.functional.linear(x, dequantize_int4_ref(qw, s, z, torch.bfloat16))
    torch.testing.assert_close(w4a16_matmul(x, qw, s, z), ref4, rtol=0, atol=0)
