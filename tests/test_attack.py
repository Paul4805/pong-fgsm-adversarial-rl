"""Tests for the FGSM core that do not require model downloads or Atari ROMs."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rai.attack import FGSMAttacker


class TinyPolicy:
    """Small deterministic policy with the PolicyAgent interface used by FGSM."""

    def __init__(self) -> None:
        torch.manual_seed(0)
        self.device = torch.device("cpu")
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(4, 8, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(16 * 9 * 9, 6),
        )
        self.net.eval()

    def tensor(self, obs: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
        tensor = torch.as_tensor(
            np.asarray(obs, dtype=np.float32)[None, ...],
            device=self.device,
            dtype=torch.float32,
        )
        if requires_grad:
            tensor.requires_grad_(True)
        return tensor

    def forward_logits(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        return self.net(obs_tensor)


@pytest.fixture()
def observation() -> np.ndarray:
    rng = np.random.default_rng(0)
    # Keep a margin from the clipping boundaries so every pixel receives the
    # full +/- epsilon FGSM step.
    return rng.uniform(0.2, 0.8, size=(4, 84, 84)).astype(np.float32)


def test_fgsm_respects_l_inf_budget(observation: np.ndarray) -> None:
    policy = TinyPolicy()
    epsilon = 0.005
    attacker = FGSMAttacker(policy, epsilon=epsilon)
    result = attacker.perturb(observation)

    assert result.adv_obs.shape == observation.shape
    assert result.adv_obs.dtype == np.float32
    assert result.adv_obs.min() >= 0.0
    assert result.adv_obs.max() <= 1.0
    assert result.max_abs_delta <= epsilon + 1e-6
    assert result.mean_abs_delta > 0.0
    assert result.grad_abs_mean > 0.0
    assert result.loss > 0.0
    assert result.gradient_action == result.label_action

    nonzero = np.abs(result.delta) > 1e-7
    assert np.allclose(np.abs(result.delta[nonzero]), epsilon, atol=1e-6)


def test_fgsm_zero_epsilon_is_identity(observation: np.ndarray) -> None:
    attacker = FGSMAttacker(TinyPolicy(), epsilon=0.0)
    result = attacker.perturb(observation)
    assert np.array_equal(result.adv_obs, observation)
    assert result.max_abs_delta == 0.0


def test_fgsm_rejects_negative_epsilon() -> None:
    with pytest.raises(ValueError):
        FGSMAttacker(TinyPolicy(), epsilon=-0.1)
