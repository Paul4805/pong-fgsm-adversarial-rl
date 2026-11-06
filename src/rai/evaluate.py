"""Evaluate clean vs. FGSM-perturbed Pong policies.

Examples
--------
White-box and black-box transfer attacks with the default RL Zoo models::

    python -m rai.evaluate --episodes 10 --epsilons 0.005 0.001

Only the white-box attack, at a single budget::

    python -m rai.evaluate --modes whitebox --epsilons 0.005
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from .attack import FGSMAttacker
from .envs import env_metadata, make_env
from .models import PROJECT_ROOT, PolicyAgent, load_agent
from .visualize import save_frame_comparison, save_summary_plots


@dataclass
class EpisodeResult:
    mode: str
    epsilon: float
    episode: int
    seed: int
    return_: float
    length: int
    steps: int
    attacked_steps: int
    changed_steps: int
    action_change_rate: float
    action_change_rate_attacked: float
    mean_abs_perturbation: float
    max_abs_perturbation: float
    mean_grad_abs: float
    mean_loss: float


def _summary(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "std": float(arr.std(ddof=0)) if arr.size else float("nan"),
        "min": float(arr.min()) if arr.size else float("nan"),
        "max": float(arr.max()) if arr.size else float("nan"),
    }


def run_episode(
    env: Any,
    target: PolicyAgent,
    seed: int,
    mode: str,
    epsilon: float,
    episode_index: int,
    attacker: FGSMAttacker | None = None,
    attack_fraction: float = 1.0,
    capture_steps: Sequence[int] = (),
    max_steps: int = 10_000,
) -> tuple[EpisodeResult, list[dict[str, Any]]]:
    """Run one episode and optionally attack at each selected step."""
    obs, _ = env.reset(seed=int(seed))
    rng = random.Random(int(seed) + 7_919)
    total_return = 0.0
    steps = 0
    attacked_steps = 0
    changed_steps = 0
    perturbation_means: list[float] = []
    perturbation_maxes: list[float] = []
    grad_means: list[float] = []
    losses: list[float] = []
    captures: list[dict[str, Any]] = []
    capture_set = set(int(s) for s in capture_steps)

    while steps < max_steps:
        clean_action = target.action(obs)
        action = clean_action
        if attacker is not None and (attack_fraction >= 1.0 or rng.random() < attack_fraction):
            attack = attacker.perturb(obs, target_clean_action=clean_action)
            adv_action = target.action(attack.adv_obs)
            attacked_steps += 1
            changed_steps += int(adv_action != clean_action)
            action = adv_action
            perturbation_means.append(attack.mean_abs_delta)
            perturbation_maxes.append(attack.max_abs_delta)
            grad_means.append(attack.grad_abs_mean)
            losses.append(attack.loss)
            if steps in capture_set:
                captures.append(
                    {
                        "step": steps,
                        "clean_obs": np.asarray(obs, dtype=np.float32).copy(),
                        "adv_obs": attack.adv_obs.copy(),
                        "clean_action": clean_action,
                        "adv_action": adv_action,
                        "epsilon": epsilon,
                    }
                )

        obs, reward, terminated, truncated, _ = env.step(action)
        total_return += float(reward)
        steps += 1
        if terminated or truncated:
            break

    result = EpisodeResult(
        mode=mode,
        epsilon=float(epsilon),
        episode=episode_index,
        seed=int(seed),
        return_=float(total_return),
        length=int(steps),
        steps=int(steps),
        attacked_steps=int(attacked_steps),
        changed_steps=int(changed_steps),
        action_change_rate=float(changed_steps / steps) if steps else 0.0,
        action_change_rate_attacked=float(changed_steps / attacked_steps)
        if attacked_steps
        else 0.0,
        mean_abs_perturbation=float(np.mean(perturbation_means)) if perturbation_means else 0.0,
        max_abs_perturbation=float(np.max(perturbation_maxes)) if perturbation_maxes else 0.0,
        mean_grad_abs=float(np.mean(grad_means)) if grad_means else 0.0,
        mean_loss=float(np.mean(losses)) if losses else 0.0,
    )
    return result, captures


def aggregate(
    mode: str,
    epsilon: float,
    attacks: Sequence[EpisodeResult],
    clean: Sequence[EpisodeResult],
    config: dict[str, Any],
) -> dict[str, Any]:
    clean_returns = [r.return_ for r in clean]
    adv_returns = [r.return_ for r in attacks]
    clean_summary = _summary(clean_returns)
    adv_summary = _summary(adv_returns)
    paired_drops = [c.return_ - a.return_ for c, a in zip(clean, attacks)]
    total_steps = sum(r.steps for r in attacks)
    total_attacked = sum(r.attacked_steps for r in attacks)
    total_changed = sum(r.changed_steps for r in attacks)
    return {
        "mode": mode,
        "epsilon": float(epsilon),
        "clean_agent": config["target"],
        "gradient_agent": config["gradient_agent"],
        "label_source": config["label_source"],
        "attack_fraction": config["attack_fraction"],
        "episodes": len(attacks),
        "clean_return_mean": clean_summary["mean"],
        "clean_return_std": clean_summary["std"],
        "adv_return_mean": adv_summary["mean"],
        "adv_return_std": adv_summary["std"],
        "reward_drop": clean_summary["mean"] - adv_summary["mean"],
        "paired_reward_drop": float(np.mean(paired_drops)),
        "reward_drop_pct": (
            100.0
            * (clean_summary["mean"] - adv_summary["mean"])
            / abs(clean_summary["mean"])
            if clean_summary["mean"] != 0
            else float("nan")
        ),
        "action_change_rate": float(total_changed / total_steps) if total_steps else 0.0,
        "action_change_rate_attacked": float(total_changed / total_attacked)
        if total_attacked
        else 0.0,
        "changed_steps": int(total_changed),
        "steps": int(total_steps),
        "attacked_steps": int(total_attacked),
        "mean_episode_length": float(np.mean([r.length for r in attacks])),
        "mean_abs_perturbation": float(np.mean([r.mean_abs_perturbation for r in attacks])),
        "max_abs_perturbation": float(np.max([r.max_abs_perturbation for r in attacks])),
        "mean_grad_abs": float(np.mean([r.mean_grad_abs for r in attacks])),
        "clean_episodes": clean_summary,
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    surrogate = summary["agents"].get("surrogate")
    surrogate_label = (
        f"**{surrogate['algorithm'].upper()}**" if surrogate else "not used"
    )
    lines = [
        "# FGSM Pong attack results",
        "",
        f"- Environment: `{summary['environment']['env_id']}` "
        f"(`{summary['environment']['ale_env_id']}` + frame skip 4)",
        f"- Target: **{summary['agents']['target']['algorithm'].upper()}**",
        f"- Black-box surrogate: {surrogate_label}",
        f"- Evaluation episodes per condition: {summary['config']['episodes']}",
        f"- Device: `{summary['config']['device']}`",
        "",
        "Action-change rate is the fraction of attacked steps where the target "
        "policy's action on the perturbed observation differs from its action "
        "on the same clean observation. The parenthesized value uses all "
        "evaluation steps as the denominator; the first value uses attacked "
        "steps only.",
        "",
        "| Mode | epsilon | Clean return | Adversarial return | Reward drop | Action change |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["attacks"]:
        lines.append(
            "| {mode} | {epsilon:g} | {clean_return_mean:.2f} | "
            "{adv_return_mean:.2f} | {reward_drop:+.2f} | "
            "{action_change_rate_attacked:.2%} ({action_change_rate:.2%}) |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Clean baseline",
            "",
            f"Mean return over {summary['clean']['episodes']} episodes: "
            f"**{summary['clean']['return']['mean']:.2f}** "
            f"(std {summary['clean']['return']['std']:.2f}).",
            "",
            "## Attack definition",
            "",
            "For every attacked step the attacker computes",
            "",
            "```",
            "loss = CrossEntropy(policy_logits(s), a_clean)",
            "s_adv = clip(s + epsilon * sign(grad_s loss), 0, 1)",
            "```",
            "",
            "and acts with the target policy on `s_adv`. For DQN the logits are "
            "Q-values; for A2C they are categorical policy logits. The white-box "
            "attack differentiates the target; the black-box attack differentiates "
            "the A2C surrogate and transfers `s_adv` to the DQN target.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="dqn", choices=["dqn", "a2c"])
    parser.add_argument("--target-checkpoint", default=None)
    parser.add_argument("--surrogate", default="a2c", choices=["dqn", "a2c"])
    parser.add_argument("--surrogate-checkpoint", default=None)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["whitebox", "blackbox"],
        choices=["whitebox", "blackbox"],
    )
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.005, 0.001])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attack-fraction", type=float, default=1.0)
    parser.add_argument("--label-source", choices=["own", "target"], default="own")
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument(
        "--capture-steps",
        nargs="+",
        type=int,
        default=[0, 50, 150, 300, 600],
    )
    parser.add_argument("--outdir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument(
        "--visdir",
        default=str(PROJECT_ROOT / "visualizations"),
    )
    parser.add_argument("--no-visualize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.attack_fraction <= 1.0:
        raise SystemExit("--attack-fraction must be in [0, 1]")
    if args.episodes < 1:
        raise SystemExit("--episodes must be positive")

    outdir = Path(args.outdir)
    visdir = Path(args.visdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not args.no_visualize:
        visdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    target = load_agent(
        args.target,
        args.target_checkpoint,
        device=args.device,
        name=f"{args.target.upper()} target",
    )
    need_surrogate = "blackbox" in args.modes
    surrogate = None
    if need_surrogate:
        surrogate = load_agent(
            args.surrogate,
            args.surrogate_checkpoint,
            device=args.device,
            name=f"{args.surrogate.upper()} surrogate",
        )
        if surrogate.algo == target.algo:
            print(
                "WARNING: surrogate and target use the same algorithm; "
                "this is not a true black-box transfer setting."
            )

    print(f"Target: {target.describe()}")
    if surrogate is not None:
        print(f"Surrogate: {surrogate.describe()}")

    env = make_env()
    env_info = env_metadata(env)
    seeds = [args.seed + i for i in range(args.episodes)]

    print(f"Running clean baseline ({args.episodes} episodes)...")
    clean_results: list[EpisodeResult] = []
    for episode, seed in enumerate(seeds):
        result, _ = run_episode(
            env=env,
            target=target,
            seed=seed,
            mode="clean",
            epsilon=0.0,
            episode_index=episode,
            attacker=None,
            max_steps=args.max_steps,
        )
        clean_results.append(result)
        print(f"  clean episode {episode + 1}/{args.episodes}: return={result.return_:.1f}")

    clean_returns = [r.return_ for r in clean_results]
    clean_summary = {
        "episodes": len(clean_results),
        "return": _summary(clean_returns),
        "length": _summary([r.length for r in clean_results]),
    }
    print(
        f"Clean mean return: {clean_summary['return']['mean']:.2f} "
        f"+/- {clean_summary['return']['std']:.2f}"
    )

    attacks: list[EpisodeResult] = []
    summary_rows: list[dict[str, Any]] = []
    attacks_by_mode: dict[str, list[EpisodeResult]] = {}

    for mode in args.modes:
        gradient_agent = target if mode == "whitebox" else surrogate
        if gradient_agent is None:
            raise RuntimeError("Black-box mode requires a surrogate")
        gradient_name = gradient_agent.name or gradient_agent.algo.upper()
        for epsilon in args.epsilons:
            attacker = FGSMAttacker(
                gradient_agent=gradient_agent,
                epsilon=epsilon,
                label_source=args.label_source,
            )
            mode_results: list[EpisodeResult] = []
            capture_records: list[dict[str, Any]] = []
            print(
                f"Running {mode} attack eps={epsilon:g} "
                f"({gradient_name} gradients -> {target.name})..."
            )
            for episode, seed in enumerate(seeds):
                result, captures = run_episode(
                    env=env,
                    target=target,
                    seed=seed,
                    mode=mode,
                    epsilon=epsilon,
                    episode_index=episode,
                    attacker=attacker,
                    attack_fraction=args.attack_fraction,
                    capture_steps=args.capture_steps if episode == 0 else (),
                    max_steps=args.max_steps,
                )
                mode_results.append(result)
                attacks.append(result)
                if episode == 0:
                    capture_records = captures
                print(
                    f"  episode {episode + 1}/{args.episodes}: return={result.return_:.1f} "
                    f"action-change={result.action_change_rate_attacked:.1%}"
                )

            config = {
                "target": target.name or target.algo.upper(),
                "gradient_agent": gradient_name,
                "label_source": args.label_source,
                "attack_fraction": args.attack_fraction,
            }
            row = aggregate(mode, epsilon, mode_results, clean_results, config)
            summary_rows.append(row)
            attacks_by_mode.setdefault(mode, []).extend(mode_results)
            print(
                f"  {mode} eps={epsilon:g}: mean return {row['adv_return_mean']:.2f}, "
                f"drop {row['reward_drop']:+.2f}, "
                f"action change {row['action_change_rate_attacked']:.1%}"
            )

            if not args.no_visualize and capture_records:
                path = visdir / f"{mode}_eps{epsilon:g}_frames.png"
                title = (
                    f"{target.name}, {mode} FGSM, epsilon={epsilon:g} "
                    f"(gradient: {gradient_name})"
                )
                save_frame_comparison(capture_records, path, title, epsilon)
                print(f"  saved visualization: {path}")

    env.close()

    if not args.no_visualize and summary_rows:
        summary_plot = visdir / "summary.png"
        save_summary_plots(summary_rows, summary_plot)
        print(f"Saved summary plot: {summary_plot}")

    summary = {
        "config": {
            "target": args.target,
            "target_checkpoint": str(target.checkpoint),
            "surrogate": args.surrogate if need_surrogate else None,
            "surrogate_checkpoint": str(surrogate.checkpoint) if surrogate else None,
            "modes": args.modes,
            "epsilons": args.epsilons,
            "episodes": args.episodes,
            "seed": args.seed,
            "device": str(target.device),
            "attack_fraction": args.attack_fraction,
            "label_source": args.label_source,
            "max_steps": args.max_steps,
        },
        "environment": env_info,
        "agents": {
            "target": target.describe(),
            "surrogate": surrogate.describe() if surrogate else None,
        },
        "clean": clean_summary,
        "attacks": summary_rows,
        "attack_episodes": [asdict(r) for r in attacks],
        "clean_episodes": [asdict(r) for r in clean_results],
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    episodes_path = outdir / "episodes.csv"
    pd.DataFrame([asdict(r) for r in clean_results + attacks]).to_csv(
        episodes_path, index=False
    )
    summary_path_csv = outdir / "summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path_csv, index=False)
    report_path = outdir / "report.md"
    _write_report(report_path, summary)

    print(f"Saved summary: {summary_path}")
    print(f"Saved per-episode data: {episodes_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
