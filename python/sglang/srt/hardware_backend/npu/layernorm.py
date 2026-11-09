# SPDX-License-Identifier: Apache-2.0
"""Shared Ascend normalization."""

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class BaseNormalization(ABC):
    @abstractmethod
    def _apply_norm(self, *args, **kwargs): ...


class NPURMSNorm(BaseNormalization):
    def _apply_norm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float | None,
        residual: torch.Tensor | None = None,
        *,
        preserve_residual_rounding: bool = False,
    ):
        if eps is None:
            eps = torch.finfo(x.dtype).eps
        if residual is not None:
            if preserve_residual_rounding:
                # Preserve the model's low-precision residual addition.
                x = residual = residual + x
            else:
                out, _, residual_out = torch.ops.npu.npu_add_rms_norm(
                    residual, x, weight, eps
                )
                return out, residual_out

        if preserve_residual_rounding and (
            x.dtype not in (torch.float16, torch.bfloat16)
            or weight.dtype != x.dtype
            or weight.device != x.device
            or x.numel() == 0
            or not hasattr(torch.ops.npu, "npu_rms_norm")
        ):
            out = F.rms_norm(x, (weight.shape[-1],), weight, eps)
        else:
            out = torch.ops.npu.npu_rms_norm(x, weight, eps)[0]
        return out if residual is None else (out, residual)
