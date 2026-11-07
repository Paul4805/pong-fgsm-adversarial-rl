"""End-to-end defense benchmark for FGSM attacks on the frozen Pong DQN.

Run everything::

    python -m rai.benchmark --stage all

Train auxiliary defenses only::

    python -m rai.benchmark --stage train

Evaluate already-trained artifacts::

    python -m rai.benchmark --stage evaluate

The benchmark deliberately keeps the original attack protocol: white-box
FGSM for A/B/C is generated from the frozen undefended DQN, while D's white-box
attack is generated from the fine-tuned robust DQN.  Black-box FGSM always uses
the A2C surrogate.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

from .attack import FGSMAttacker
from .defenses.autoencoder import (
    ConvAutoencoder,
    DenoiseAutoencoder,
    calibrate_threshold,
    load_autoencoder,
    reconstruction_mse,
    save_autoencoder,
    train_autoencoder,
)
from .defenses.data import CleanRolloutData, collect_clean_data, data_summary
from .defenses.robust_dqn import fine_tune_dqn
from .defenses.vq import PatchVQ, train_vq
from .envs import env_metadata, make_env
from .models import PROJECT_ROOT, PolicyAgent, load_agent, rescale_model_to_unit_interval

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "defense_benchmark"
RESULT_DIR = PROJECT_ROOT / "results" / "defense_benchmark"
PLOT_DIR = RESULT_DIR / "plots"
FRAME_DIR = RESULT_DIR / "frames"

CONDITIONS = [
    {"name": "clean", "attack_type": "clean", "epsilon": 0.0},
    {"name": "whitebox_0.001", "attack_type": "whitebox", "epsilon": 0.001},
    {"name": "whitebox_0.005", "attack_type": "whitebox", "epsilon": 0.005},
    {"name": "blackbox_0.005", "attack_type": "blackbox", "epsilon": 0.005},
]

DEFENSE_LABELS = {
    "baseline": "Undefended DQN",
    "A_autoencoder": "A: Denoising AE",
    "B_detect_denoise": "B: Detect + denoise",
    "C_vector_quantization": "C: Vector quantization",
    "D_adversarial_training": "D: Adversarial fine-tune",
}


@dataclass
class DefenseEpisodeResult:
    defense: str
    condition: str
    attack_type: str
    epsilon: float
    episode: int
    seed: int
    return_: float
    length: int
    steps: int
    changed_steps: int
    action_change_rate: float
    flagged_steps: int
    flagged_rate: float
    mean_abs_perturbation: float
    max_abs_perturbation: float


@dataclass
class DefenseSpec:
    name: str
    agent: PolicyAgent
    transform: Callable[[np.ndarray], tuple[np.ndarray, dict[str, Any]]]
    description: str
    params: int
    train_seconds: float
    extra_latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _identity_transform(obs: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    return np.asarray(obs, dtype=np.float32), {"flagged": False}


def _autoencoder_transform(denoiser: DenoiseAutoencoder):
    def transform(obs: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        return denoiser.transform(obs), {"flagged": False}

    return transform


def _detect_denoise_transform(denoiser: DenoiseAutoencoder, threshold: float):
    def transform(obs: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        recon, error = denoiser.process(obs)
        flagged = bool(error > threshold)
        # Reuse the reconstruction already computed for the detector when the
        # sample is flagged; this avoids a second autoencoder forward pass.
        return (recon if flagged else np.asarray(obs, dtype=np.float32)), {
            "flagged": flagged,
            "reconstruction_error": error,
        }

    return transform


def _vq_transform(vq: PatchVQ):
    def transform(obs: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        return vq.quantize(obs), {"flagged": False}

    return transform


def _measure_latency(
    transform: Callable[[np.ndarray], tuple[np.ndarray, dict[str, Any]]],
    obs: np.ndarray,
    device: torch.device,
    repeats: int = 30,
) -> float:
    for _ in range(3):
        transform(obs)
    _sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        transform(obs)
    _sync(device)
    return (time.perf_counter() - start) / repeats * 1000.0


def _evaluate_clean_return(
    agent: PolicyAgent,
    episodes: int,
    seed: int,
    max_steps: int = 10_000,
) -> float:
    env = make_env()
    returns: list[float] = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total = 0.0
        steps = 0
        done = False
        while not done and steps < max_steps:
            action = agent.action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            steps += 1
            done = bool(terminated or truncated)
        returns.append(total)
    env.close()
    return float(np.mean(returns))


def _resolve_device_arg(device_arg: str) -> torch.device:
    if str(device_arg) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "train", "evaluate"], default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=10_000)

    parser.add_argument("--data-episodes", type=int, default=12)
    parser.add_argument("--max-transitions", type=int, default=6000)
    parser.add_argument("--observation-stride", type=int, default=4)

    parser.add_argument("--ae-epochs", type=int, default=12)
    parser.add_argument("--ae-batch-size", type=int, default=128)
    parser.add_argument("--ae-lr", type=float, default=1e-3)
    parser.add_argument("--ae-noise-std", type=float, default=0.01)
    parser.add_argument("--ae-sign-noise-eps", type=float, default=0.005)

    parser.add_argument("--vq-k", type=int, default=128)
    parser.add_argument("--vq-patch-size", type=int, default=6)
    parser.add_argument("--vq-iterations", type=int, default=25)
    parser.add_argument("--vq-max-patches", type=int, default=200_000)

    parser.add_argument("--detector-fpr", type=float, default=0.01)
    parser.add_argument("--recon-sample-size", type=int, default=128)

    parser.add_argument("--d-epochs", type=int, default=8)
    parser.add_argument("--d-batch-size", type=int, default=64)
    parser.add_argument("--d-lr", type=float, default=1e-4)
    parser.add_argument("--d-adv-fraction", type=float, default=1.0)
    parser.add_argument("--d-loss-mode", choices=["margin", "distill", "td"], default="margin")
    parser.add_argument("--d-eps", nargs="+", type=float, default=[0.005, 0.01])
    parser.add_argument("--d-margin", type=float, default=1.0)
    parser.add_argument("--d-eval-episodes", type=int, default=3)

    parser.add_argument("--capture-steps", nargs="+", type=int, default=[0, 50, 150, 300, 600])
    parser.add_argument("--artifacts", default=str(ARTIFACT_DIR))
    parser.add_argument("--results", default=str(RESULT_DIR))
    parser.add_argument("--no-visualize", action="store_true")
    return parser.parse_args()


def _artifact_paths(artifact_dir: Path) -> dict[str, Path]:
    return {
        "autoencoder": artifact_dir / "defense_a_autoencoder.pt",
        "vq": artifact_dir / "defense_c_vq.npz",
        "robust_dqn": artifact_dir / "defense_d_robust_dqn.zip",
        "config": artifact_dir / "benchmark_config.json",
        "timings": artifact_dir / "timings.json",
        "metadata": artifact_dir / "defense_metadata.json",
    }


def train_artifacts(args: argparse.Namespace, artifact_dir: Path) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(artifact_dir)
    device = _resolve_device_arg(args.device)

    print("=" * 78)
    print("Stage 1/3: collecting clean data and training auxiliary defenses")
    print("=" * 78)
    baseline = load_agent("dqn", device=device, name="baseline DQN")
    data = collect_clean_data(
        baseline,
        episodes=args.data_episodes,
        seed=args.seed,
        max_transitions=args.max_transitions,
        observation_stride=args.observation_stride,
        max_steps=args.max_steps,
    )

    timings: dict[str, float] = {"data_collection_seconds": data.collection_seconds}

    # ---- Defense A / B: denoising autoencoder + detector -----------------
    print("\nTraining Defense A autoencoder...")
    ae_start = time.perf_counter()
    ae, ae_meta = train_autoencoder(
        data.frames,
        device=device,
        epochs=args.ae_epochs,
        batch_size=args.ae_batch_size,
        learning_rate=args.ae_lr,
        noise_std=args.ae_noise_std,
        sign_noise_eps=args.ae_sign_noise_eps,
        seed=args.seed,
    )
    timings["autoencoder_train_seconds"] = time.perf_counter() - ae_start
    val_start = int(len(data.frames) * 0.9)
    val_frames = data.frames[val_start:]
    clean_recon_mse = reconstruction_mse(ae, val_frames, device)
    denoiser = DenoiseAutoencoder(model=ae, device=device)
    clean_errors = np.asarray(
        [denoiser.process(obs.astype(np.float32) / 255.0)[1] for obs in data.observations],
        dtype=np.float64,
    )
    threshold = calibrate_threshold(clean_errors, args.detector_fpr)

    attacker = FGSMAttacker(baseline, epsilon=0.005)
    n_recon = min(args.recon_sample_size, len(data.observations))
    adv_obs_samples: list[np.ndarray] = []
    adv_errors: list[float] = []
    for obs_u8 in data.observations[:n_recon]:
        clean_obs = obs_u8.astype(np.float32) / 255.0
        adv_obs = attacker.perturb(clean_obs).adv_obs
        adv_obs_samples.append(adv_obs)
        adv_errors.append(denoiser.process(adv_obs)[1])
    adv_obs_array = np.stack(adv_obs_samples, axis=0)
    adv_frames = np.rint(adv_obs_array[:, -1] * 255.0).clip(0, 255).astype(np.uint8)
    adv_recon_mse = reconstruction_mse(ae, adv_frames, device)
    ae_meta.update(
        {
            "clean_reconstruction_mse": clean_recon_mse,
            "adversarial_reconstruction_mse": adv_recon_mse,
            "detector_threshold": threshold,
            "detector_fpr_target": args.detector_fpr,
            "detector_fpr_calibration": float(np.mean(clean_errors > threshold)),
            "detector_tpr_sample": float(np.mean(np.asarray(adv_errors) > threshold)),
            "calibration_observations": int(len(clean_errors)),
            "reconstruction_sample_size": int(n_recon),
        }
    )
    save_autoencoder(ae, paths["autoencoder"], ae_meta)
    print(
        f"  AE: clean MSE={clean_recon_mse:.6f}, adv MSE={adv_recon_mse:.6f}, "
        f"threshold={threshold:.6f}, sample TPR={ae_meta['detector_tpr_sample']:.2%}"
    )

    # ---- Defense C: vector quantization ------------------------------------
    print("\nTraining Defense C vector quantizer...")
    vq_start = time.perf_counter()
    codebook, vq_meta = train_vq(
        data.frames,
        device=device,
        k=args.vq_k,
        patch_size=args.vq_patch_size,
        max_patches=args.vq_max_patches,
        iterations=args.vq_iterations,
        seed=args.seed,
    )
    timings["vq_train_seconds"] = time.perf_counter() - vq_start
    vq = PatchVQ(
        codebook=codebook,
        device=device,
        patch_size=args.vq_patch_size,
        k=args.vq_k,
    )
    vq_meta["clean_distortion_mse"] = vq.distortion(val_frames)
    vq_meta["clean_codebook_utilization"] = vq.utilization(val_frames)
    vq.save(paths["vq"], vq_meta)
    print(
        f"  VQ: K={args.vq_k}, patch={args.vq_patch_size}, "
        f"clean distortion MSE={vq_meta['clean_distortion_mse']:.6f}, "
        f"utilization={vq_meta['clean_codebook_utilization']:.1%}"
    )

    # ---- Defense D: adversarial fine-tuning --------------------------------
    print("\nFine-tuning Defense D DQN...")
    d_start = time.perf_counter()
    robust_agent = load_agent("dqn", device=device, name="D robust DQN")
    d_result = fine_tune_dqn(
        robust_agent,
        data,
        epochs=args.d_epochs,
        batch_size=args.d_batch_size,
        learning_rate=args.d_lr,
        adversarial_fraction=args.d_adv_fraction,
        seed=args.seed,
        loss_mode=args.d_loss_mode,
        teacher_agent=baseline,
        epsilon_choices=args.d_eps,
        margin=args.d_margin,
        clean_eval_fn=lambda agent, episodes: _evaluate_clean_return(
            agent, episodes, seed=args.seed + 10_000, max_steps=args.max_steps
        ),
        clean_eval_episodes=args.d_eval_episodes,
    )
    timings["adversarial_finetune_seconds"] = time.perf_counter() - d_start
    # Save the robust checkpoint using the original uint8 observation-space
    # metadata so SB3 can rebuild NatureCNN on load.  The first convolution
    # already contains the [0,1] rescaling, so evaluation loads it with
    # already_normalized=True and does not rescale again.
    normalized_space = d_result.model.observation_space
    raw_space = spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
    d_result.model.observation_space = raw_space
    d_result.model.policy.observation_space = raw_space
    d_result.model.save(str(paths["robust_dqn"]))
    d_result.model.observation_space = normalized_space
    d_result.model.policy.observation_space = normalized_space
    # Verify the saved checkpoint is reloadable with the normalized input
    # convention before the evaluation stage.
    _ = load_agent(
        "dqn",
        paths["robust_dqn"],
        device=device,
        name="D robust DQN",
        already_normalized=True,
    )
    d_meta = d_result.metadata
    d_meta["checkpoint"] = str(paths["robust_dqn"])
    d_meta["final_clean_return"] = _evaluate_clean_return(
        robust_agent, args.d_eval_episodes, seed=args.seed + 10_000, max_steps=args.max_steps
    )
    print(f"  D: final clean return={d_meta['final_clean_return']:.2f}")

    metadata = {
        "config": vars(args),
        "data": data_summary(data),
        "autoencoder": ae_meta,
        "vq": vq_meta,
        "robust_dqn": d_meta,
        "timings": timings,
        "environment": {},
    }
    meta_env = make_env()
    metadata["environment"] = env_metadata(meta_env)
    meta_env.close()
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    paths["timings"].write_text(json.dumps(timings, indent=2), encoding="utf-8")
    paths["config"].write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"\nSaved artifacts to {artifact_dir}")
    return metadata


def _load_defenses(args: argparse.Namespace, artifact_dir: Path) -> tuple[list[DefenseSpec], dict[str, Any], PolicyAgent, PolicyAgent]:
    paths = _artifact_paths(artifact_dir)
    device = _resolve_device_arg(args.device)
    required = [paths["autoencoder"], paths["vq"], paths["robust_dqn"], paths["metadata"]]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing defense artifacts. Run `python -m rai.benchmark --stage train` first. "
            f"Missing: {missing}"
        )
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    baseline = load_agent("dqn", device=device, name="baseline DQN")
    surrogate = load_agent("a2c", device=device, name="A2C surrogate")
    ae, ae_meta = load_autoencoder(paths["autoencoder"], device)
    denoiser = DenoiseAutoencoder(ae, device)
    threshold = float(ae_meta["detector_threshold"])
    vq, vq_meta = PatchVQ.load(paths["vq"], device)
    robust = load_agent(
        "dqn",
        paths["robust_dqn"],
        device=device,
        name="D robust DQN",
        already_normalized=True,
    )

    ae_params = int(sum(p.numel() for p in ae.parameters()))
    defenses = [
        DefenseSpec(
            name="baseline",
            agent=baseline,
            transform=_identity_transform,
            description="Frozen pre-trained DQN, no input defense.",
            params=0,
            train_seconds=0.0,
            metadata={},
        ),
        DefenseSpec(
            name="A_autoencoder",
            agent=baseline,
            transform=_autoencoder_transform(denoiser),
            description="All observations are reconstructed by the clean-trained denoising AE.",
            params=ae_params,
            train_seconds=float(metadata["timings"]["autoencoder_train_seconds"]),
            metadata=ae_meta,
        ),
        DefenseSpec(
            name="B_detect_denoise",
            agent=baseline,
            transform=_detect_denoise_transform(denoiser, threshold),
            description="Reconstruction-error detector; flagged observations are denoised.",
            params=ae_params,
            train_seconds=float(metadata["timings"]["autoencoder_train_seconds"]),
            metadata={**ae_meta, "detector_threshold": threshold},
        ),
        DefenseSpec(
            name="C_vector_quantization",
            agent=baseline,
            transform=_vq_transform(vq),
            description=f"6x6 patch VQ with K={args.vq_k} codes trained on clean frames.",
            params=int(vq_meta["parameters"]),
            train_seconds=float(metadata["timings"]["vq_train_seconds"]),
            metadata=vq_meta,
        ),
        DefenseSpec(
            name="D_adversarial_training",
            agent=robust,
            transform=_identity_transform,
            description="DQN fine-tuned on mixed clean and FGSM-perturbed transitions.",
            params=int(sum(p.numel() for p in robust.model.policy.parameters())),
            train_seconds=float(metadata["timings"]["adversarial_finetune_seconds"]),
            metadata=metadata["robust_dqn"],
        ),
    ]
    return defenses, metadata, baseline, surrogate


def _run_episode(
    env: Any,
    defense: DefenseSpec,
    baseline_agent: PolicyAgent,
    attacker: FGSMAttacker | None,
    seed: int,
    condition: str,
    attack_type: str,
    epsilon: float,
    episode_index: int,
    capture_steps: Sequence[int] = (),
    max_steps: int = 10_000,
) -> tuple[DefenseEpisodeResult, list[dict[str, Any]]]:
    obs, _ = env.reset(seed=int(seed))
    captures: list[dict[str, Any]] = []
    capture_set = set(int(s) for s in capture_steps)
    total_return = 0.0
    steps = 0
    changed_steps = 0
    flagged_steps = 0
    perturbation_means: list[float] = []
    perturbation_maxes: list[float] = []
    done = False

    while not done and steps < max_steps:
        # Reference action from the clean, undefended baseline policy on the
        # current state.  This is the denominator used for action-change rate.
        reference_action = baseline_agent.action(obs)

        raw_obs = np.asarray(obs, dtype=np.float32)
        if attacker is not None:
            attack = attacker.perturb(raw_obs)
            raw_obs = attack.adv_obs
            perturbation_means.append(attack.mean_abs_delta)
            perturbation_maxes.append(attack.max_abs_delta)

        defended_obs, info = defense.transform(raw_obs)
        action = defense.agent.action(defended_obs)
        changed_steps += int(action != reference_action)
        if bool(info.get("flagged", False)):
            flagged_steps += 1

        if steps in capture_set:
            captures.append(
                {
                    "step": steps,
                    "clean_obs": np.asarray(obs, dtype=np.float32).copy(),
                    "attack_obs": np.asarray(raw_obs, dtype=np.float32).copy(),
                    "defended_obs": np.asarray(defended_obs, dtype=np.float32).copy(),
                    "reference_action": int(reference_action),
                    "defended_action": int(action),
                    "flagged": bool(info.get("flagged", False)),
                }
            )

        obs, reward, terminated, truncated, _ = env.step(action)
        total_return += float(reward)
        steps += 1
        done = bool(terminated or truncated)

    result = DefenseEpisodeResult(
        defense=defense.name,
        condition=condition,
        attack_type=attack_type,
        epsilon=float(epsilon),
        episode=episode_index,
        seed=int(seed),
        return_=float(total_return),
        length=int(steps),
        steps=int(steps),
        changed_steps=int(changed_steps),
        action_change_rate=float(changed_steps / steps) if steps else 0.0,
        flagged_steps=int(flagged_steps),
        flagged_rate=float(flagged_steps / steps) if steps else 0.0,
        mean_abs_perturbation=float(np.mean(perturbation_means)) if perturbation_means else 0.0,
        max_abs_perturbation=float(np.max(perturbation_maxes)) if perturbation_maxes else 0.0,
    )
    return result, captures


def _save_frame_comparison(
    captures: Sequence[dict[str, Any]],
    path: Path,
    title: str,
    defense_name: str,
) -> None:
    if not captures:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(captures)
    fig, axes = plt.subplots(n, 3, figsize=(9.0, 2.75 * n), squeeze=False, constrained_layout=True)
    for row, record in enumerate(captures):
        clean = record["clean_obs"][-1]
        attacked = record["attack_obs"][-1]
        defended = record["defended_obs"][-1]
        axes[row, 0].imshow(clean, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 1].imshow(attacked, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 2].imshow(defended, cmap="gray", vmin=0.0, vmax=1.0)
        if row == 0:
            axes[row, 0].set_title("Clean", fontsize=10)
            axes[row, 1].set_title("FGSM-perturbed", fontsize=10)
            if defense_name == "D_adversarial_training":
                axes[row, 2].set_title("Defended input (identity)", fontsize=10)
            else:
                axes[row, 2].set_title("Post-defense input", fontsize=10)
        axes[row, 0].set_ylabel(
            f"step {record['step']}\n"
            f"a={record['reference_action']} -> {record['defended_action']}\n"
            f"flag={int(record['flagged'])}",
            fontsize=8,
        )
        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_defense_results(
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    plot_dir: Path,
    metadata: dict[str, Any],
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    defenses = long_df["defense"].drop_duplicates().tolist()
    labels = {d: DEFENSE_LABELS.get(d, d) for d in defenses}
    attack_conditions = ["whitebox_0.001", "whitebox_0.005", "blackbox_0.005"]
    condition_labels = {
        "clean": "Clean",
        "whitebox_0.001": "WB ε=0.001",
        "whitebox_0.005": "WB ε=0.005",
        "blackbox_0.005": "BB ε=0.005",
    }

    # Reward by defense and condition.
    fig, ax = plt.subplots(figsize=(12.5, 5.2), constrained_layout=True)
    pivot = long_df.pivot(index="condition", columns="defense", values="return_mean").reindex(
        ["clean", "whitebox_0.001", "whitebox_0.005", "blackbox_0.005"]
    )
    pivot = pivot[[d for d in defenses if d in pivot.columns]]
    x = np.arange(len(pivot.index))
    width = 0.8 / max(len(pivot.columns), 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(pivot.columns)))
    for i, defense in enumerate(pivot.columns):
        ax.bar(x + (i - (len(pivot.columns) - 1) / 2) * width, pivot[defense], width, label=labels[defense], color=colors[i])
    ax.set_xticks(x, [condition_labels[c] for c in pivot.index])
    ax.set_ylabel("Mean episode return")
    ax.set_title("Clean and attacked returns by defense")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.savefig(plot_dir / "reward_by_condition.png", dpi=180)
    plt.close(fig)

    # Reward drop relative to the clean undefended baseline, attack only.
    fig, ax = plt.subplots(figsize=(12.0, 4.8), constrained_layout=True)
    drop_pivot = long_df[long_df["condition"].isin(attack_conditions)].pivot(
        index="condition", columns="defense", values="reward_drop"
    ).reindex(attack_conditions)
    drop_pivot = drop_pivot[[d for d in defenses if d in drop_pivot.columns]]
    x = np.arange(len(drop_pivot.index))
    width = 0.8 / max(len(drop_pivot.columns), 1)
    for i, defense in enumerate(drop_pivot.columns):
        ax.bar(x + (i - (len(drop_pivot.columns) - 1) / 2) * width, drop_pivot[defense], width, label=labels[defense], color=colors[i])
    ax.set_xticks(x, [condition_labels[c] for c in drop_pivot.index])
    ax.set_ylabel("Reward drop vs. clean undefended baseline")
    ax.set_title("Attack-induced reward drop by defense")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.savefig(plot_dir / "reward_drop_by_condition.png", dpi=180)
    plt.close(fig)

    # Action change.
    fig, ax = plt.subplots(figsize=(12.0, 4.8), constrained_layout=True)
    change_pivot = long_df[long_df["condition"].isin(attack_conditions)].pivot(
        index="condition", columns="defense", values="action_change_rate"
    ).reindex(attack_conditions)
    change_pivot = change_pivot[[d for d in defenses if d in change_pivot.columns]]
    for i, defense in enumerate(change_pivot.columns):
        ax.bar(
            x + (i - (len(change_pivot.columns) - 1) / 2) * width,
            change_pivot[defense] * 100.0,
            width,
            label=labels[defense],
            color=colors[i],
        )
    ax.set_xticks(x, [condition_labels[c] for c in change_pivot.index])
    ax.set_ylabel("Action-change rate (%)")
    ax.set_title("Action changes relative to the clean undefended DQN")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.savefig(plot_dir / "action_change_by_condition.png", dpi=180)
    plt.close(fig)

    # Pareto frontier: robustness vs. inference and training cost.
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True)
    x_inf = wide_df["extra_latency_ms"].clip(lower=0.0) + 0.01
    y = wide_df["mean_attack_return"]
    axes[0].scatter(x_inf, y, s=80, c=np.arange(len(wide_df)), cmap="tab10")
    for i, row in wide_df.iterrows():
        axes[0].annotate(labels.get(row["defense"], row["defense"]), (x_inf.iloc[i], y.iloc[i]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Additional inference latency (ms/step, log scale)")
    axes[0].set_ylabel("Mean return under attack (higher = more robust)")
    axes[0].set_title("Robustness vs. inference cost")

    x_train = wide_df["train_seconds"].clip(lower=0.0) + 1.0
    axes[1].scatter(x_train, y, s=80, c=np.arange(len(wide_df)), cmap="tab10")
    for i, row in wide_df.iterrows():
        axes[1].annotate(labels.get(row["defense"], row["defense"]), (x_train.iloc[i], y.iloc[i]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Auxiliary training time (seconds, log scale; baseline=1s offset)")
    axes[1].set_ylabel("Mean return under attack (higher = more robust)")
    axes[1].set_title("Robustness vs. training cost")
    fig.savefig(plot_dir / "pareto_robustness_cost.png", dpi=180)
    plt.close(fig)

    # Clean vs. mean-attack return.
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    order = wide_df["defense"].tolist()
    x = np.arange(len(order))
    ax.bar(x - 0.18, wide_df["clean_return"], 0.36, label="Clean", color="#4C78A8")
    ax.bar(x + 0.18, wide_df["mean_attack_return"], 0.36, label="Mean under attack", color="#E45756")
    ax.set_xticks(x, [labels[d] for d in order], rotation=20, ha="right")
    ax.set_ylabel("Mean episode return")
    ax.set_title("Clean performance vs. mean attacked return")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.legend(frameon=False)
    fig.savefig(plot_dir / "clean_vs_mean_attack.png", dpi=180)
    plt.close(fig)

    # Autoencoder training curve.
    ae_hist = metadata.get("autoencoder", {}).get("history", [])
    if ae_hist:
        fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
        epochs = [row["epoch"] for row in ae_hist]
        ax.plot(epochs, [row["train_loss"] for row in ae_hist], label="train loss")
        ax.plot(epochs, [row["val_clean_mse"] for row in ae_hist], label="val clean MSE")
        ax.plot(epochs, [row["val_noisy_mse"] for row in ae_hist], label="val noisy MSE")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_title("Defense A autoencoder training")
        ax.legend(frameon=False)
        fig.savefig(plot_dir / "autoencoder_training.png", dpi=180)
        plt.close(fig)


def _write_summary_csvs(
    results: Sequence[DefenseEpisodeResult],
    metadata: dict[str, Any],
    result_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows: list[dict[str, Any]] = []
    defenses = list(DEFENSE_LABELS.keys())
    baseline_clean = float(
        np.mean([r.return_ for r in results if r.defense == "baseline" and r.condition == "clean"])
    )

    for defense in defenses:
        for condition in CONDITIONS:
            rows = [
                r
                for r in results
                if r.defense == defense and r.condition == condition["name"]
            ]
            if not rows:
                continue
            returns = np.asarray([r.return_ for r in rows], dtype=np.float64)
            steps = sum(r.steps for r in rows)
            changed = sum(r.changed_steps for r in rows)
            flagged = sum(r.flagged_steps for r in rows)
            long_rows.append(
                {
                    "defense": defense,
                    "defense_label": DEFENSE_LABELS[defense],
                    "condition": condition["name"],
                    "attack_type": condition["attack_type"],
                    "epsilon": condition["epsilon"],
                    "episodes": len(rows),
                    "return_mean": float(returns.mean()),
                    "return_std": float(returns.std(ddof=0)),
                    "return_min": float(returns.min()),
                    "return_max": float(returns.max()),
                    "reward_drop": baseline_clean - float(returns.mean()),
                    "action_change_rate": float(changed / steps) if steps else 0.0,
                    "changed_steps": int(changed),
                    "steps": int(steps),
                    "flagged_steps": int(flagged),
                    "flagged_rate": float(flagged / steps) if steps else 0.0,
                    "mean_abs_perturbation": float(np.mean([r.mean_abs_perturbation for r in rows])),
                    "max_abs_perturbation": float(np.max([r.max_abs_perturbation for r in rows])),
                }
            )

    long_df = pd.DataFrame(long_rows)
    # Recovery relative to the undefended attack result.
    for condition in ["whitebox_0.001", "whitebox_0.005", "blackbox_0.005"]:
        base = long_df[(long_df.defense == "baseline") & (long_df.condition == condition)]
        if base.empty:
            continue
        base_return = float(base.iloc[0]["return_mean"])
        denom = baseline_clean - base_return
        for idx, row in long_df[long_df.condition == condition].iterrows():
            if abs(denom) > 1e-9:
                long_df.loc[idx, "recovery_fraction"] = (row["return_mean"] - base_return) / denom
            else:
                long_df.loc[idx, "recovery_fraction"] = float("nan")

    wide_rows: list[dict[str, Any]] = []
    for defense in defenses:
        d_long = long_df[long_df.defense == defense]
        if d_long.empty:
            continue
        row: dict[str, Any] = {"defense": defense, "defense_label": DEFENSE_LABELS[defense]}
        for condition in ["clean", "whitebox_0.001", "whitebox_0.005", "blackbox_0.005"]:
            hit = d_long[d_long.condition == condition]
            if hit.empty:
                continue
            rec = hit.iloc[0]
            row[f"{condition}_return"] = rec["return_mean"]
            row[f"{condition}_std"] = rec["return_std"]
            row[f"{condition}_drop"] = rec["reward_drop"]
            row[f"{condition}_action_change"] = rec["action_change_rate"]
            row[f"{condition}_flagged_rate"] = rec["flagged_rate"]
            if "recovery_fraction" in rec and not pd.isna(rec["recovery_fraction"]):
                row[f"{condition}_recovery"] = rec["recovery_fraction"]
        attacks = d_long[d_long.condition != "clean"]
        row["mean_attack_return"] = float(attacks["return_mean"].mean())
        row["mean_attack_drop"] = baseline_clean - row["mean_attack_return"]
        row["mean_action_change"] = float(
            attacks["changed_steps"].sum() / max(attacks["steps"].sum(), 1)
        )
        row["clean_return"] = row.get("clean_return", float("nan"))
        row["clean_drop"] = row.get("clean_drop", float("nan"))
        row["extra_latency_ms"] = float(
            next((d.extra_latency_ms for d in metadata["_defenses"] if d.name == defense), 0.0)
        )
        row["train_seconds"] = float(
            next((d.train_seconds for d in metadata["_defenses"] if d.name == defense), 0.0)
        )
        row["params"] = int(next((d.params for d in metadata["_defenses"] if d.name == defense), 0))
        row["detection_fpr"] = (
            float(row.get("clean_flagged_rate", float("nan"))) if defense == "B_detect_denoise" else float("nan")
        )
        row["detection_tpr_wb_0.005"] = (
            float(row.get("whitebox_0.005_flagged_rate", float("nan")))
            if defense == "B_detect_denoise"
            else float("nan")
        )
        row["detection_tpr_bb_0.005"] = (
            float(row.get("blackbox_0.005_flagged_rate", float("nan")))
            if defense == "B_detect_denoise"
            else float("nan")
        )
        wide_rows.append(row)

    wide_df = pd.DataFrame(wide_rows)
    long_df.to_csv(result_dir / "summary_long.csv", index=False)
    wide_df.to_csv(result_dir / "summary.csv", index=False)
    pd.DataFrame([asdict(r) for r in results]).to_csv(result_dir / "episodes.csv", index=False)
    return wide_df, long_df


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_report(
    wide_df: pd.DataFrame,
    long_df: pd.DataFrame,
    metadata: dict[str, Any],
    result_dir: Path,
) -> None:
    baseline_clean = float(wide_df.loc[wide_df.defense == "baseline", "clean_return"].iloc[0])
    lines = [
        "# Defense benchmark report",
        "",
        "## Setup",
        "",
        "- Environment: `PongNoFrameskip-v4` equivalent (`ALE/Pong-v5`, frame skip 4, "
        "grayscale, 84×84, 4-frame stack, `[0,1]` normalization).",
        "- Target: pre-trained RL Zoo DQN. The target remains frozen except for Defense D.",
        "- Surrogate: pre-trained RL Zoo A2C, used for black-box transfer.",
        "- Attack: FGSM with L∞ budgets 0.001 and 0.005; 10 paired evaluation seeds per condition.",
        f"- Clean undefended baseline return: **{baseline_clean:.2f}**.",
        "",
        "## Main results",
        "",
        "| Defense | Clean | WB 0.001 | WB 0.005 | BB 0.005 | Mean attack drop | Action change | Extra latency | Train time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in wide_df.iterrows():
        lines.append(
            f"| {row['defense_label']} | {_fmt(row['clean_return'])} | "
            f"{_fmt(row['whitebox_0.001_return'])} | {_fmt(row['whitebox_0.005_return'])} | "
            f"{_fmt(row['blackbox_0.005_return'])} | {_fmt(row['mean_attack_drop'])} | "
            f"{_fmt(100.0 * row['mean_action_change'], 1)}% | "
            f"{_fmt(row['extra_latency_ms'], 2)} ms | {_fmt(row['train_seconds'], 1)} s |"
        )
    lines.extend(["", "## Long-form metrics", "", "```", long_df.to_string(index=False), "```", ""])

    ae = metadata.get("autoencoder", {})
    vq = metadata.get("vq", {})
    rd = metadata.get("robust_dqn", {})
    b = wide_df[wide_df.defense == "B_detect_denoise"]
    lines.extend(
        [
            "## Defense-specific metrics",
            "",
            "### A — Denoising autoencoder",
            "",
            f"- Architecture: 4-layer convolutional encoder / decoder, "
            f"{ae.get('parameters', 0):,} parameters.",
            f"- Training: {ae.get('epochs')} epochs, batch size {ae.get('batch_size')}, "
            f"lr {ae.get('learning_rate')}, Gaussian σ={ae.get('noise_std')}, "
            f"sign-noise ε={ae.get('sign_noise_eps')}.",
            f"- Clean reconstruction MSE: **{_fmt(ae.get('clean_reconstruction_mse'), 6)}**.",
            f"- Adversarial reconstruction MSE: **{_fmt(ae.get('adversarial_reconstruction_mse'), 6)}**.",
            "",
            "### B — Detect and denoise",
            "",
            f"- Detector: mean per-frame reconstruction error, threshold calibrated to "
            f"{100 * ae.get('detector_fpr_target', 0):.0f}% clean FPR.",
            f"- Threshold: **{_fmt(ae.get('detector_threshold'), 6)}**.",
            f"- Calibration FPR: {100 * ae.get('detector_fpr_calibration', 0):.2f}%; "
            f"sample TPR at ε=0.005: **{100 * ae.get('detector_tpr_sample', 0):.2f}%**.",
        ]
    )
    if not b.empty:
        lines.append(
            f"- Evaluation FPR on clean episodes: {100 * b.iloc[0]['detection_fpr']:.2f}%; "
            f"TPR on white-box ε=0.005: {100 * b.iloc[0]['detection_tpr_wb_0.005']:.2f}%; "
            f"TPR on black-box ε=0.005: {100 * b.iloc[0]['detection_tpr_bb_0.005']:.2f}%."
        )
    lines.extend(
        [
            "",
            "### C — Vector quantization",
            "",
            f"- Codebook: K={vq.get('k')} 6×6 patch codes ({vq.get('parameters', 0):,} floats).",
            f"- Clean-frame quantization distortion: **{_fmt(vq.get('clean_distortion_mse'), 6)} MSE**.",
            f"- Codebook utilization on clean validation patches: "
            f"**{100 * vq.get('clean_codebook_utilization', 0):.2f}%**.",
            "",
            "### D — Adversarial fine-tuning",
            "",
            f"- Offline FGSM-augmented fine-tuning: {rd.get('epochs')} epochs, batch size "
            f"{rd.get('batch_size')}, lr {rd.get('learning_rate')}, adversarial fraction "
            f"{rd.get('adversarial_fraction')}.",
            f"- Transitions: {rd.get('transitions')}, update steps: {rd.get('global_steps')}.",
            f"- Training time: {_fmt(rd.get('train_seconds'), 1)} s; peak VRAM: "
            f"{_fmt(rd.get('max_vram_mb'), 0)} MB.",
            "",
            "## Compute time",
            "",
            "| Stage | Seconds |",
            "|---|---:|",
        ]
    )
    for key, value in metadata.get("timings", {}).items():
        lines.append(f"| {key} | {value:.1f} |")
    lines.extend(
        [
            "",
            "## Critical analysis",
            "",
            "**A — Denoising autoencoder.** The autoencoder is the lowest-complexity "
            "input defense, but it has no knowledge of the policy or of the attack. "
            "Its reconstruction loss on clean vs. adversarial frames quantifies whether "
            "the bottleneck removes perturbation energy; the reward results show whether "
            "that removal preserves task-relevant paddle/ball information. A denoiser "
            "that over-smooths the ball or paddle can reduce clean performance and may "
            "still fail under adaptive attacks that optimize through it.",
            "",
            "**B — Detect and denoise.** The detector is calibrated for a controlled "
            "clean FPR, so its main trade-off is detection sensitivity vs. clean "
            "intervention. A low TPR means most attacked frames bypass denoising; a high "
            "TPR with a noisy detector can excessively denoise clean gameplay. Because "
            "the detector and denoiser share one reconstruction forward pass, its "
            "inference overhead is close to Defense A. Detection is statistical rather "
            "than certified and can be evaded by perturbations that keep reconstruction "
            "error below threshold.",
            "",
            "**C — Vector quantization.** Patch-wise VQ constrains the input to a finite "
            "set of clean-frame patch prototypes. This is cheap and attack-agnostic, but "
            "it introduces block quantization artifacts. The measured clean distortion "
            "and codebook utilization show how much information is lost: using too few "
            "codes can erase the ball, while many rarely used codes provide little "
            "defense. VQ is also non-differentiable in the standard nearest-neighbor "
            "form, so adaptive white-box attacks require a straight-through or "
            "gradient-free approximation.",
            "",
            "**D — Adversarial fine-tuning.** This is the only defense that changes the "
            "policy and therefore has the strongest chance of preserving task-relevant "
            "features under attack. The cost is a clean-performance trade-off and a "
            "training-time cost; the benchmark records both. The fine-tuning is offline "
            "and lightweight (a few epochs on a moderate replay sample), not a full "
            "RS-DQN student/Q-network training procedure. It can also overfit the "
            "specific FGSM budgets used during training, so adaptive or larger-budget "
            "attacks should be tested separately.",
            "",
            "**Pareto comparison.** Defense D is expected to occupy the high-robustness / "
            "high-training-cost end, while A/B/C are cheap at inference. Defense B can "
            "dominate A when the detector is both sensitive and precise, because it "
            "leaves most clean frames untouched. The plots in `plots/` show the empirical "
            "frontier; no single defense dominates on all axes.",
            "",
            "## Limitations",
            "",
            "- White-box attacks for A/B/C are generated from the frozen undefended DQN, "
            "matching the established benchmark protocol. They are not adaptive to the "
            "defense transform; an adaptive attack through the autoencoder/VQ could be "
            "stronger.",
            "- Defense D's white-box attack is generated from the fine-tuned DQN, so it "
            "is adaptive to that policy.",
            "- The benchmark uses deterministic greedy actions and 10 episodes per "
            "condition; variance is reported in the CSVs.",
            "- Pong is a low-dimensional, visually simple game; conclusions may not "
            "transfer to richer Atari games.",
            "",
        ]
    )
    (result_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate_artifacts(args: argparse.Namespace, artifact_dir: Path, result_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = result_dir / "plots"
    frame_dir = result_dir / "frames"
    defenses, metadata, baseline, surrogate = _load_defenses(args, artifact_dir)
    device = baseline.device

    # Measure transform latency on a clean observation, except for identity
    # defenses where there is zero additional transform cost.
    env = make_env()
    sample_obs, _ = env.reset(seed=args.seed)
    adv_sample = FGSMAttacker(baseline, epsilon=0.005).perturb(sample_obs).adv_obs
    env.close()
    for defense in defenses:
        if defense.name in {"baseline", "D_adversarial_training"}:
            defense.extra_latency_ms = 0.0
        else:
            defense.extra_latency_ms = _measure_latency(defense.transform, adv_sample, device)
    metadata["_defenses"] = defenses

    results: list[DefenseEpisodeResult] = []
    evaluation_start = time.perf_counter()
    env = make_env()
    for defense in defenses:
        defense_start = time.perf_counter()
        for condition in CONDITIONS:
            if condition["attack_type"] == "whitebox":
                attacker = FGSMAttacker(defense.agent, epsilon=condition["epsilon"])
            elif condition["attack_type"] == "blackbox":
                attacker = FGSMAttacker(surrogate, epsilon=condition["epsilon"])
            else:
                attacker = None

            print(
                f"\nEvaluating {DEFENSE_LABELS[defense.name]} | {condition['name']} "
                f"({args.episodes} episodes)"
            )
            captures: list[dict[str, Any]] = []
            for episode in range(args.episodes):
                seed = args.seed + episode
                capture_steps = (
                    args.capture_steps
                    if episode == 0 and condition["name"] == "whitebox_0.005" and defense.name != "baseline"
                    else ()
                )
                result, episode_captures = _run_episode(
                    env=env,
                    defense=defense,
                    baseline_agent=baseline,
                    attacker=attacker,
                    seed=seed,
                    condition=condition["name"],
                    attack_type=condition["attack_type"],
                    epsilon=condition["epsilon"],
                    episode_index=episode,
                    capture_steps=capture_steps,
                    max_steps=args.max_steps,
                )
                results.append(result)
                if capture_steps:
                    captures = episode_captures
                print(
                    f"  episode {episode + 1:02d}/{args.episodes}: "
                    f"return={result.return_:.1f}, action-change={result.action_change_rate:.1%}"
                )
            if captures and not args.no_visualize:
                path = frame_dir / f"{defense.name}_whitebox_eps0.005_frames.png"
                _save_frame_comparison(
                    captures,
                    path,
                    f"{DEFENSE_LABELS[defense.name]} · white-box FGSM ε=0.005",
                    defense.name,
                )
                print(f"  saved frames: {path}")
        defense_seconds = time.perf_counter() - defense_start
        metadata.setdefault("_evaluation_seconds", {})[defense.name] = defense_seconds
        print(f"  {DEFENSE_LABELS[defense.name]} evaluation time: {defense_seconds:.1f}s")
    env.close()
    metadata["evaluation_seconds"] = time.perf_counter() - evaluation_start

    wide_df, long_df = _write_summary_csvs(results, metadata, result_dir)
    _plot_defense_results(long_df, wide_df, plot_dir, metadata)
    _write_report(wide_df, long_df, metadata, result_dir)

    clean_meta = {
        "config": vars(args),
        "data": metadata.get("data", {}),
        "autoencoder": metadata.get("autoencoder", {}),
        "vq": metadata.get("vq", {}),
        "robust_dqn": metadata.get("robust_dqn", {}),
        "timings": {
            **metadata.get("timings", {}),
            "evaluation_seconds": metadata["evaluation_seconds"],
            "evaluation_by_defense_seconds": metadata.get("_evaluation_seconds", {}),
        },
        "defenses": {
            d.name: {
                "description": d.description,
                "params": d.params,
                "extra_latency_ms": d.extra_latency_ms,
                "train_seconds": d.train_seconds,
            }
            for d in defenses
        },
        "environment": metadata.get("environment", {}),
    }
    (result_dir / "benchmark_metadata.json").write_text(
        json.dumps(clean_meta, indent=2), encoding="utf-8"
    )
    print(f"\nSaved benchmark results to {result_dir}")
    print(f"Summary: {result_dir / 'summary.csv'}")
    print(f"Report:  {result_dir / 'report.md'}")


def main() -> None:
    args = _parse_args()
    artifact_dir = Path(args.artifacts)
    result_dir = Path(args.results)
    if args.stage in {"all", "train"}:
        train_artifacts(args, artifact_dir)
    if args.stage in {"all", "evaluate"}:
        evaluate_artifacts(args, artifact_dir, result_dir)


if __name__ == "__main__":
    main()
