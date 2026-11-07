"""Clean-data collection shared by the autoencoder, VQ, and robust-DQN stages."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..envs import make_env
from ..models import PolicyAgent


def _to_uint8_obs(obs: np.ndarray) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32)
    return np.rint(arr * 255.0).clip(0, 255).astype(np.uint8)


@dataclass
class CleanRolloutData:
    frames: np.ndarray
    observations: np.ndarray
    transition_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    episodes: int
    env_steps: int
    collection_seconds: float


def collect_clean_data(
    agent: PolicyAgent,
    episodes: int = 12,
    seed: int = 2026,
    max_transitions: int = 6000,
    observation_stride: int = 4,
    max_steps: int = 10_000,
    verbose: bool = True,
) -> CleanRolloutData:
    """Roll out the frozen policy and collect clean frames/transitions.

    ``frames`` are used to train the autoencoder and VQ codebook.
    ``observations`` are used to calibrate the detector threshold.
    ``transitions`` are used for lightweight adversarial fine-tuning.
    """
    env = make_env()
    frames: list[np.ndarray] = []
    observations: list[np.ndarray] = []
    transition_observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    next_observations: list[np.ndarray] = []
    dones: list[bool] = []
    start_time = time.perf_counter()

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        steps = 0
        done = False
        while not done and steps < max_steps:
            frames.append(_to_uint8_obs(obs[-1]))
            if steps % observation_stride == 0:
                observations.append(_to_uint8_obs(obs))
            action = agent.action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            if len(actions) < max_transitions:
                transition_observations.append(_to_uint8_obs(obs))
                actions.append(action)
                rewards.append(float(reward))
                next_observations.append(_to_uint8_obs(next_obs))
                dones.append(bool(terminated or truncated))
            obs = next_obs
            steps += 1
            done = bool(terminated or truncated)
        if verbose:
            print(
                f"  data episode {episode + 1:02d}/{episodes}: "
                f"steps={steps}, transitions={len(actions)}"
            )

    env.close()
    collection_seconds = time.perf_counter() - start_time
    data = CleanRolloutData(
        frames=np.stack(frames, axis=0).astype(np.uint8),
        observations=np.stack(observations, axis=0).astype(np.uint8),
        transition_observations=np.stack(transition_observations, axis=0).astype(np.uint8),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        next_observations=np.stack(next_observations, axis=0).astype(np.uint8),
        dones=np.asarray(dones, dtype=np.bool_),
        episodes=episodes,
        env_steps=len(frames),
        collection_seconds=collection_seconds,
    )
    if verbose:
        print(
            f"  collected {len(data.frames)} frames, {len(data.observations)} stacked obs, "
            f"{len(data.actions)} transitions in {collection_seconds:.1f}s"
        )
    return data


def data_summary(data: CleanRolloutData) -> dict[str, Any]:
    return {
        "episodes": data.episodes,
        "frames": int(len(data.frames)),
        "observations": int(len(data.observations)),
        "transitions": int(len(data.actions)),
        "env_steps": int(data.env_steps),
        "collection_seconds": data.collection_seconds,
    }
