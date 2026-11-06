"""Policy checkpoint loading, compatibility, and normalized-input adapters."""

from __future__ import annotations

import json
import shutil
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from huggingface_hub import hf_hub_download
from stable_baselines3 import A2C, DQN

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"

HF_REPOS = {
    "dqn": ("sb3/dqn-PongNoFrameskip-v4", "dqn-PongNoFrameskip-v4.zip"),
    "a2c": ("sb3/a2c-PongNoFrameskip-v4", "a2c-PongNoFrameskip-v4.zip"),
}

# The RL Zoo checkpoints were serialized with older SB3 versions.  These
# replacements are only used to deserialize the schedule callables; they do not
# affect the network weights used by the attack.
CUSTOM_OBJECTS = {
    "learning_rate": 1e-4,
    "lr_schedule": lambda _: 1e-4,
    "exploration_schedule": lambda _: 0.01,
}


def default_checkpoint(algo: str) -> Path:
    return DEFAULT_MODEL_DIR / algo / "model.zip"


def download_pretrained(force: bool = False) -> dict[str, Path]:
    """Download the RL Zoo DQN and A2C Pong checkpoints if needed."""
    paths: dict[str, Path] = {}
    for algo, (repo_id, filename) in HF_REPOS.items():
        target_dir = DEFAULT_MODEL_DIR / algo
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "model.zip"
        if force or not target.exists():
            downloaded = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(target_dir),
                )
            )
            if downloaded.resolve() != target.resolve():
                shutil.copy2(downloaded, target)
                downloaded.unlink(missing_ok=True)
        # Keep the corresponding RL Zoo config for provenance.
        config = target_dir / "config.yml"
        if force or not config.exists():
            downloaded_cfg = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename="config.yml",
                    local_dir=str(target_dir),
                )
            )
            if downloaded_cfg.resolve() != config.resolve():
                shutil.copy2(downloaded_cfg, config)
                downloaded_cfg.unlink(missing_ok=True)
        paths[algo] = target
    return paths


def ensure_compat_checkpoint(path: str | Path) -> Path:
    """Patch an old DQN archive so SB3 >= 2.9 can deserialize it.

    The RL Zoo DQN was saved with ``optimize_memory_usage=True`` and a replay
    buffer configuration that newer SB3 rejects before the policy is even used.
    The replay buffer is irrelevant for FGSM evaluation, so the archive's JSON
    config is rewritten with ``optimize_memory_usage=False``.
    """
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        data = json.loads(archive.read("data"))
    if not data.get("optimize_memory_usage", False):
        return path

    compat = path.with_name(f"{path.stem}_compat.zip")
    if compat.exists() and compat.stat().st_mtime >= path.stat().st_mtime:
        return compat

    tmp = compat.with_suffix(".data.tmp")
    data["optimize_memory_usage"] = False
    # A full buffer is unnecessary for loading/evaluation.
    data["buffer_size"] = max(int(data.get("buffer_size", 1)), 10_000)
    tmp.write_text(json.dumps(data), encoding="utf-8")
    try:
        with zipfile.ZipFile(path) as source, zipfile.ZipFile(
            compat, "w", compression=zipfile.ZIP_DEFLATED
        ) as dest:
            for item in source.infolist():
                if item.filename == "data":
                    dest.write(tmp, "data")
                else:
                    dest.writestr(item, source.read(item.filename))
    finally:
        tmp.unlink(missing_ok=True)
    return compat


def _first_conv(module: torch.nn.Module) -> torch.nn.Conv2d | None:
    for child in module.modules():
        if isinstance(child, torch.nn.Conv2d):
            return child
    return None


def _feature_extractors(policy: Any, algo: str) -> list[torch.nn.Module]:
    modules: list[torch.nn.Module] = []
    if algo == "dqn":
        q_net = getattr(policy, "q_net", None)
        if q_net is not None:
            modules.append(q_net.features_extractor)
    else:
        for attr in ("features_extractor", "pi_features_extractor", "vf_features_extractor"):
            module = getattr(policy, attr, None)
            if module is not None:
                modules.append(module)
    # Shared actor/critic extractors appear multiple times.
    unique: list[torch.nn.Module] = []
    seen: set[int] = set()
    for module in modules:
        if id(module) not in seen:
            seen.add(id(module))
            unique.append(module)
    return unique


def rescale_model_to_unit_interval(model: Any, algo: str, scale: float = 255.0) -> None:
    """Make a model trained on [0, 255] accept [0, 1] observations exactly.

    For a first convolution ``y = W x + b``, feeding ``x/255`` is equivalent to
    feeding ``x`` to a network whose first-layer weights are ``255 * W``.  This
    folds the normalization into the checkpoint rather than silently
    multiplying inputs inside the attack code.
    """
    if getattr(model, "_rai_unit_interval_input", False):
        return
    for extractor in _feature_extractors(model.policy, algo):
        conv = _first_conv(extractor)
        if conv is None:
            raise RuntimeError(f"Could not find a convolution in {type(extractor).__name__}")
        with torch.no_grad():
            conv.weight.mul_(scale)

    obs_shape = model.observation_space.shape
    normalized_space = spaces.Box(low=0.0, high=1.0, shape=obs_shape, dtype=np.float32)
    model.observation_space = normalized_space
    model.policy.observation_space = normalized_space
    model._rai_unit_interval_input = True


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass
class PolicyAgent:
    """Uniform Q-value / policy-logit interface for DQN and A2C."""

    model: Any
    algo: str
    checkpoint: Path
    device: torch.device
    name: str | None = None

    def __post_init__(self) -> None:
        self.model.policy.eval()
        if self.name is None:
            self.name = self.algo.upper()

    def tensor(self, obs: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
        arr = np.asarray(obs, dtype=np.float32)
        tensor = torch.as_tensor(arr[None, ...], device=self.device, dtype=torch.float32)
        if requires_grad:
            tensor.requires_grad_(True)
        return tensor

    def forward_logits(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Return action logits (A2C) or Q-values (DQN)."""
        if obs_tensor.ndim != 4:
            raise ValueError(f"Expected (B, C, H, W) observations, got {tuple(obs_tensor.shape)}")
        if self.algo == "dqn":
            return self.model.policy.q_net(obs_tensor)
        distribution = self.model.policy.get_distribution(obs_tensor)
        # SB3 wraps torch.distributions.Categorical in CategoricalDistribution.
        inner = getattr(distribution, "distribution", distribution)
        logits = getattr(inner, "logits", None)
        if logits is None:
            raise RuntimeError("Could not extract logits from the A2C policy distribution")
        return logits

    @torch.no_grad()
    def logits(self, obs: np.ndarray) -> torch.Tensor:
        return self.forward_logits(self.tensor(obs)).squeeze(0)

    @torch.no_grad()
    def action(self, obs: np.ndarray) -> int:
        return int(self.logits(obs).argmax().item())

    @torch.no_grad()
    def action_values(self, obs: np.ndarray) -> np.ndarray:
        return self.logits(obs).detach().cpu().numpy()

    def describe(self) -> dict[str, Any]:
        n_params = sum(p.numel() for p in self.model.policy.parameters())
        return {
            "name": self.name,
            "algorithm": self.algo,
            "checkpoint": str(self.checkpoint),
            "device": str(self.device),
            "parameters": int(n_params),
            "observation_space": str(self.model.observation_space),
            "action_space": str(self.model.action_space),
            "unit_interval_input": bool(getattr(self.model, "_rai_unit_interval_input", False)),
            "num_timesteps": int(getattr(self.model, "num_timesteps", 0)),
        }


def load_agent(
    algo: str,
    checkpoint: str | Path | None = None,
    device: str | torch.device | None = "auto",
    name: str | None = None,
) -> PolicyAgent:
    """Load a DQN/A2C checkpoint and prepare it for normalized observations."""
    algo = algo.lower()
    if algo not in {"dqn", "a2c"}:
        raise ValueError(f"Unsupported algorithm: {algo}")
    path = Path(checkpoint) if checkpoint is not None else default_checkpoint(algo)
    if not path.exists():
        download_pretrained()
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Run `python -m rai.download_models` "
            "or pass an existing --target-checkpoint/--surrogate-checkpoint."
        )
    if algo == "dqn":
        path = ensure_compat_checkpoint(path)
    resolved_device = _resolve_device(device)
    cls = DQN if algo == "dqn" else A2C
    with warnings.catch_warnings():
        # The RL Zoo archives pre-date several SB3 serialization changes.  The
        # warnings are expected and the weights load correctly.
        warnings.filterwarnings("ignore", message=".*OpenAI Gym.*")
        warnings.filterwarnings("ignore", message=".*saved with SB3 <.*")
        warnings.filterwarnings("ignore", message=".*optimizer state.*")
        model = cls.load(
            str(path),
            device=resolved_device,
            custom_objects=CUSTOM_OBJECTS,
        )
    rescale_model_to_unit_interval(model, algo)
    return PolicyAgent(
        model=model,
        algo=algo,
        checkpoint=path,
        device=resolved_device,
        name=name,
    )
