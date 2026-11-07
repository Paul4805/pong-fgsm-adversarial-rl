"""Patch-wise vector quantization input transformation (Defense C)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _frames_to_tensor(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    arr = np.asarray(frames)
    if arr.dtype == np.uint8:
        tensor = torch.from_numpy(arr).to(device=device, dtype=torch.float32).div_(255.0)
    else:
        tensor = torch.as_tensor(arr, device=device, dtype=torch.float32)
    return tensor.unsqueeze(1)


def _extract_patches(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    # x: (B, 1, H, W) -> (B * L, patch_size**2)
    patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)
    return patches.transpose(1, 2).reshape(-1, patch_size * patch_size)


def _quantize_tensor(
    x: torch.Tensor,
    codebook: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    # x: (B, 1, H, W)
    patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)
    flat = patches.transpose(1, 2)  # (B, L, p*p)
    distances = torch.cdist(flat, codebook)
    indices = distances.argmin(dim=-1)
    quantized = codebook[indices]  # (B, L, p*p)
    quantized = quantized.transpose(1, 2)
    out = F.fold(
        quantized,
        output_size=(x.shape[-2], x.shape[-1]),
        kernel_size=patch_size,
        stride=patch_size,
    )
    return out.clamp_(0.0, 1.0)


def train_vq(
    frames: np.ndarray,
    device: torch.device,
    k: int = 128,
    patch_size: int = 6,
    max_patches: int = 200_000,
    iterations: int = 25,
    chunk_size: int = 4096,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train a patch codebook with mini-batch k-means on clean frames."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if 84 % patch_size != 0:
        raise ValueError("patch_size must divide 84 to avoid padding artifacts")

    start_time = time.perf_counter()
    x = _frames_to_tensor(frames, device)
    patches = _extract_patches(x, patch_size)
    if len(patches) > max_patches:
        idx = torch.randperm(len(patches), device=device)[:max_patches]
        patches = patches[idx]
    if len(patches) < k:
        raise ValueError(f"Need at least {k} patches, got {len(patches)}")

    centers = patches[torch.randperm(len(patches), device=device)[:k]].clone()
    for iteration in range(iterations):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(k, device=device)
        for start in range(0, len(patches), chunk_size):
            chunk = patches[start : start + chunk_size]
            distances = torch.cdist(chunk, centers)
            assignment = distances.argmin(dim=-1)
            sums.index_add_(0, assignment, chunk)
            counts.index_add_(0, assignment, torch.ones_like(assignment, dtype=torch.float32))
        new_centers = sums / counts.clamp_min(1.0).unsqueeze(1)
        empty = counts == 0
        if bool(empty.any()):
            n_empty = int(empty.sum().item())
            replacement = patches[torch.randperm(len(patches), device=device)[:n_empty]]
            new_centers[empty] = replacement
        shift = float(torch.linalg.vector_norm(new_centers - centers).item())
        centers = new_centers
        if verbose:
            print(f"  VQ k-means iteration {iteration + 1:02d}/{iterations}: center shift={shift:.6f}")
        if shift < 1e-5:
            break

    # Final assignment distortion and utilization.
    distortion_total = 0.0
    count = 0
    used: set[int] = set()
    for start in range(0, len(patches), chunk_size):
        chunk = patches[start : start + chunk_size]
        distances = torch.cdist(chunk, centers)
        assignment = distances.argmin(dim=-1)
        chosen = centers[assignment]
        distortion_total += float(((chunk - chosen) ** 2).sum().item())
        count += int(chunk.numel())
        used.update(int(v) for v in assignment.detach().cpu().tolist())

    train_seconds = time.perf_counter() - start_time
    metadata = {
        "k": k,
        "patch_size": patch_size,
        "max_patches": max_patches,
        "iterations": iterations,
        "train_seconds": train_seconds,
        "patch_distortion_mse": distortion_total / max(count, 1),
        "codebook_utilization": len(used) / k,
        "used_codes": len(used),
        "total_patches_trained": int(len(patches)),
        "parameters": int(k * patch_size * patch_size),
    }
    return centers.detach().cpu().numpy().astype(np.float32), metadata


@dataclass
class PatchVQ:
    """Quantize every 6x6 patch of a stacked observation to a codebook entry."""

    codebook: np.ndarray
    device: torch.device
    patch_size: int = 6
    k: int = 128

    def __post_init__(self) -> None:
        self._codebook = torch.as_tensor(self.codebook, device=self.device, dtype=torch.float32)

    def quantize(self, obs: np.ndarray) -> np.ndarray:
        obs_arr = np.asarray(obs, dtype=np.float32)
        with torch.no_grad():
            x = torch.as_tensor(obs_arr[None], device=self.device, dtype=torch.float32)
            frames = x.reshape(-1, 1, 84, 84)
            quantized = _quantize_tensor(frames, self._codebook, self.patch_size)
            return quantized.reshape(4, 84, 84).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def distortion(self, frames: np.ndarray, batch_size: int = 256) -> float:
        total = 0.0
        count = 0
        for start in range(0, len(frames), batch_size):
            x = _frames_to_tensor(frames[start : start + batch_size], self.device)
            q = _quantize_tensor(x, self._codebook, self.patch_size)
            total += float(F.mse_loss(q, x, reduction="sum").item())
            count += int(x.numel())
        return total / max(count, 1)

    @torch.no_grad()
    def utilization(self, frames: np.ndarray, batch_size: int = 256) -> float:
        used: set[int] = set()
        for start in range(0, len(frames), batch_size):
            x = _frames_to_tensor(frames[start : start + batch_size], self.device)
            patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
            flat = patches.transpose(1, 2).reshape(-1, self.patch_size * self.patch_size)
            distances = torch.cdist(flat, self._codebook)
            assignment = distances.argmin(dim=-1)
            used.update(int(v) for v in assignment.detach().cpu().tolist())
        return len(used) / self.k

    def save(self, path: str | Path, metadata: dict[str, Any]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            codebook=self.codebook,
            patch_size=np.asarray(self.patch_size),
            k=np.asarray(self.k),
            metadata=np.asarray(json.dumps(metadata)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path, device: torch.device) -> tuple["PatchVQ", dict[str, Any]]:
        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data["metadata"].item()))
        return (
            cls(
                codebook=np.asarray(data["codebook"], dtype=np.float32),
                device=device,
                patch_size=int(data["patch_size"]),
                k=int(data["k"]),
            ),
            metadata,
        )
