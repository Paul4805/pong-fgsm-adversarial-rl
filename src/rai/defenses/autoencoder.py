"""Standalone convolutional denoising autoencoder (Defense A and B component)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ConvAutoencoder(nn.Module):
    """Small convolutional autoencoder for 84x84 grayscale Pong frames.

    The input is reflect-padded to 88x88 so three stride-2 convolutions produce
    an 11x11 latent map and the decoder can return the original 84x84 size.
    """

    def __init__(self, latent_channels: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (2, 2, 2, 2), mode="reflect")
        z = self.encoder(x)
        out = self.decoder(z)
        return out[:, :, 2:86, 2:86]


def _frames_to_tensor(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    arr = np.asarray(frames)
    if arr.dtype == np.uint8:
        tensor = torch.from_numpy(arr).to(device=device, dtype=torch.float32).div_(255.0)
    else:
        tensor = torch.as_tensor(arr, device=device, dtype=torch.float32)
    return tensor.unsqueeze(1)


@torch.no_grad()
def reconstruction_mse(
    model: ConvAutoencoder,
    frames: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for start in range(0, len(frames), batch_size):
        x = _frames_to_tensor(frames[start : start + batch_size], device)
        recon = model(x)
        total += float(F.mse_loss(recon, x, reduction="sum").item())
        count += int(x.numel())
    return total / max(count, 1)


def train_autoencoder(
    frames: np.ndarray,
    device: torch.device,
    epochs: int = 12,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    noise_std: float = 0.01,
    sign_noise_eps: float = 0.005,
    validation_fraction: float = 0.1,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[ConvAutoencoder, dict[str, Any]]:
    """Train a denoising autoencoder on clean frames.

    The targets are always clean frames.  Inputs are mixed between clean frames,
    Gaussian noise, and FGSM-like sign noise.  This is a standard denoising
    objective and keeps the autoencoder independent of the RL policy.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ConvAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n = len(frames)
    if n < 100:
        raise ValueError("Need at least 100 clean frames to train the autoencoder")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * validation_fraction))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_frames = np.asarray(frames)[train_idx]
    val_frames = np.asarray(frames)[val_idx]

    history: list[dict[str, float]] = []
    start_time = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_frames))
        running = 0.0
        batches = 0
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            clean = _frames_to_tensor(train_frames[idx], device)
            corrupt = clean.clone()
            # Half Gaussian, half sign-pattern noise.  The sign noise mimics the
            # L_inf FGSM structure while keeping the clean target.
            mode = torch.rand(len(clean), 1, 1, 1, device=device)
            gaussian = torch.randn_like(clean) * noise_std
            sign = torch.sign(torch.randn_like(clean))
            sign = sign * (torch.rand_like(clean) * sign_noise_eps)
            corrupt = torch.where(mode < 0.5, clean + gaussian, clean + sign)
            corrupt = corrupt.clamp_(0.0, 1.0)

            recon = model(corrupt)
            loss = F.mse_loss(recon, clean)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            batches += 1

        val_clean = reconstruction_mse(model, val_frames, device, batch_size=batch_size)
        # Validate the denoising objective on a held-out noisy copy.
        noisy = val_frames.copy()
        noisy_float = noisy.astype(np.float32) / 255.0
        noisy_float = np.clip(
            noisy_float + rng.normal(0.0, noise_std, size=noisy_float.shape), 0.0, 1.0
        )
        noisy_u8 = np.rint(noisy_float * 255.0).astype(np.uint8)
        val_noisy = reconstruction_mse(model, noisy_u8, device, batch_size=batch_size)
        record = {
            "epoch": epoch + 1,
            "train_loss": running / max(batches, 1),
            "val_clean_mse": val_clean,
            "val_noisy_mse": val_noisy,
        }
        history.append(record)
        if verbose:
            print(
                f"  AE epoch {epoch + 1:02d}/{epochs}: "
                f"train={record['train_loss']:.6f} "
                f"val_clean={val_clean:.6f} val_noisy={val_noisy:.6f}"
            )

    train_seconds = time.perf_counter() - start_time
    metadata = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "noise_std": noise_std,
        "sign_noise_eps": sign_noise_eps,
        "train_frames": int(len(train_frames)),
        "val_frames": int(len(val_frames)),
        "train_seconds": train_seconds,
        "history": history,
        "parameters": int(sum(p.numel() for p in model.parameters())),
    }
    return model, metadata


def save_autoencoder(
    model: ConvAutoencoder,
    path: str | Path,
    metadata: dict[str, Any],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
    return path


def load_autoencoder(path: str | Path, device: torch.device) -> tuple[ConvAutoencoder, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ConvAutoencoder()
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    metadata = json.loads(json.dumps(checkpoint.get("metadata", {})))
    return model, metadata


@dataclass
class DenoiseAutoencoder:
    """Per-frame denoising wrapper used at evaluation time."""

    model: ConvAutoencoder
    device: torch.device

    def reconstruct(self, obs: np.ndarray) -> np.ndarray:
        obs_arr = np.asarray(obs, dtype=np.float32)
        with torch.no_grad():
            x = torch.as_tensor(obs_arr[None], device=self.device, dtype=torch.float32)
            frames = x.reshape(-1, 1, 84, 84)
            recon = self.model(frames).clamp_(0.0, 1.0)
            return recon.reshape(4, 84, 84).cpu().numpy().astype(np.float32)

    def process(self, obs: np.ndarray) -> tuple[np.ndarray, float]:
        obs_arr = np.asarray(obs, dtype=np.float32)
        recon = self.reconstruct(obs_arr)
        error = float(np.mean((obs_arr - recon) ** 2))
        return recon, error

    def transform(self, obs: np.ndarray) -> np.ndarray:
        return self.reconstruct(obs)


def calibrate_threshold(
    errors: np.ndarray,
    false_positive_rate: float = 0.05,
) -> float:
    """Threshold that flags the largest ``false_positive_rate`` of clean errors."""
    if not 0.0 < false_positive_rate < 1.0:
        raise ValueError("false_positive_rate must be in (0, 1)")
    return float(np.quantile(np.asarray(errors, dtype=np.float64), 1.0 - false_positive_rate))
