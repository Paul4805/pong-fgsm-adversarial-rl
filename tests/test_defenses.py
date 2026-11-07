"""Unit tests for the auxiliary defense modules."""

from __future__ import annotations

import numpy as np
import torch

from rai.defenses.autoencoder import (
    ConvAutoencoder,
    DenoiseAutoencoder,
    calibrate_threshold,
    reconstruction_mse,
)
from rai.defenses.robust_dqn import _fgsm_batch
from rai.defenses.vq import PatchVQ


def test_autoencoder_shape_and_range() -> None:
    model = ConvAutoencoder()
    x = torch.rand(2, 1, 84, 84)
    with torch.no_grad():
        out = model(x)
    assert out.shape == x.shape
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_denoiser_returns_stacked_observation() -> None:
    model = ConvAutoencoder()
    denoiser = DenoiseAutoencoder(model, torch.device("cpu"))
    obs = np.random.default_rng(0).random((4, 84, 84), dtype=np.float32)
    recon, error = denoiser.process(obs)
    assert recon.shape == obs.shape
    assert recon.dtype == np.float32
    assert np.isfinite(error)
    assert error >= 0.0


def test_threshold_calibration_hits_requested_quantile() -> None:
    errors = np.arange(100, dtype=np.float64)
    threshold = calibrate_threshold(errors, false_positive_rate=0.05)
    assert 94.0 <= threshold <= 95.0
    assert np.mean(errors > threshold) <= 0.05 + 1e-9


def test_patch_vq_shape_and_distortion() -> None:
    rng = np.random.default_rng(1)
    codebook = rng.random((16, 36), dtype=np.float32)
    vq = PatchVQ(codebook=codebook, device=torch.device("cpu"), patch_size=6, k=16)
    obs = rng.random((4, 84, 84), dtype=np.float32)
    q = vq.quantize(obs)
    assert q.shape == obs.shape
    assert q.dtype == np.float32
    assert q.min() >= 0.0
    assert q.max() <= 1.0
    assert np.isfinite(vq.distortion(obs[-1:]))
    assert 0.0 <= vq.utilization(obs[-1:]) <= 1.0


def test_reconstruction_mse_perfect_model() -> None:
    # A model that returns its input should have zero reconstruction error.
    class Identity(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    frames = np.random.default_rng(2).random((8, 84, 84), dtype=np.float32)
    assert reconstruction_mse(Identity(), frames, torch.device("cpu")) < 1e-12


def test_training_fgsm_batch_respects_budget() -> None:
    q_net = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(4 * 84 * 84, 6),
    )
    obs = torch.as_tensor(
        np.random.default_rng(3).uniform(0.2, 0.8, size=(2, 4, 84, 84)),
        dtype=torch.float32,
    )
    actions = torch.tensor([0, 5], dtype=torch.long)
    epsilon = 0.005
    adv = _fgsm_batch(q_net, obs, actions, epsilon)
    assert adv.shape == obs.shape
    assert float((adv - obs).abs().max()) <= epsilon + 1e-6
    assert float(adv.min()) >= 0.0
    assert float(adv.max()) <= 1.0
