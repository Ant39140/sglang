# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
import torch

import sglang.multimodal_gen.runtime.platforms as platforms
from sglang.multimodal_gen.runtime.layers import layernorm
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.platforms.interface import Platform, PlatformEnum


class _OotPlatform(Platform):
    _enum = PlatformEnum.OOT

    def get_dispatch_key_name(self):
        return "test"


@pytest.mark.parametrize("preserve", [False, True])
def test_rmsnorm_reference_rounding_with_registered_oot_kernel(monkeypatch, preserve):
    platform = _OotPlatform()
    monkeypatch.setattr(platforms, "_current_platform", platform)
    monkeypatch.setattr(layernorm, "current_platform", platform)
    monkeypatch.setattr(CustomOp, "_oot_forward_registry", {})
    value = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16)
    plugin = MagicMock(return_value=torch.zeros_like(value))
    CustomOp.register_oot_forward(RMSNorm, fn=plugin, platform_key="test")
    norm = RMSNorm(4, eps=1e-6, preserve_residual_rounding=preserve).to(value.dtype)

    actual = norm(value)

    if preserve:
        reference = torch.nn.RMSNorm(4, eps=1e-6, dtype=value.dtype)
        torch.testing.assert_close(actual, reference(value), rtol=0, atol=0)
        plugin.assert_not_called()
    else:
        torch.testing.assert_close(actual, torch.zeros_like(value))
        plugin.assert_called_once_with(norm, value)
