import torch
import torch.nn as nn

from .gemm import w4a16_matmul
from .quantize import dequantize_int4_ref, quantize_int4


class Int4Linear(nn.Module):
    """Drop-in replacement for nn.Linear with int4 group-quantized weights (W4A16)."""

    def __init__(self, in_features: int, out_features: int, group_size: int = 128, bias: bool = False, device=None):
        super().__init__()
        assert in_features % group_size == 0
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        n_groups = in_features // group_size
        self.register_buffer("qweight", torch.empty((in_features // 8, out_features), dtype=torch.int32, device=device))
        self.register_buffer("scales", torch.empty((n_groups, out_features), dtype=torch.float16, device=device))
        self.register_buffer("zeros", torch.empty((n_groups, out_features), dtype=torch.float16, device=device))
        self.register_buffer("bias", torch.empty(out_features, device=device) if bias else None)

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear: nn.Linear, group_size: int = 128) -> "Int4Linear":
        w = linear.weight.data
        m = cls(linear.in_features, linear.out_features, group_size, linear.bias is not None, device=w.device)
        m.qweight, m.scales, m.zeros = quantize_int4(w.contiguous(), group_size)
        if linear.bias is not None:
            m.bias = linear.bias.data.clone()
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = w4a16_matmul(x, self.qweight, self.scales, self.zeros)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y

    def dequantize(self, dtype=torch.bfloat16) -> torch.Tensor:
        return dequantize_int4_ref(self.qweight, self.scales, self.zeros, dtype)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, group_size={self.group_size}"


def _iter_linears(model: nn.Module, skip):
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and not any(s in full for s in skip):
                yield module, child_name, child


@torch.no_grad()
def quantize_model_linears(model: nn.Module, group_size: int = 128, skip=("lm_head",), fake: bool = False) -> nn.Module:
    """Replace every nn.Linear (except names containing `skip`) with Int4Linear.

    fake=True keeps nn.Linear but overwrites its weight with the dequantized int4 weight
    (reference for checking that the Triton path reproduces the same model quality).
    """
    for parent, child_name, child in list(_iter_linears(model, skip)):
        q = Int4Linear.from_linear(child, group_size)
        if fake:
            child.weight.data.copy_(q.dequantize(child.weight.dtype))
        else:
            setattr(parent, child_name, q)
        torch.cuda.empty_cache()
    return model
