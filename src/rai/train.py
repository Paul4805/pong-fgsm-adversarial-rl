"""Optional from-scratch Pong training.

The reproduction in this repository uses the public RL Zoo checkpoints because
they are already trained to a stable positive Pong return.  This script is
provided so the same pipeline can be trained locally if desired.

Models are trained on the standard raw uint8 observations in [0, 255].
``rai.models.load_agent`` folds the /255 normalization into the first
convolution, so the attack/evaluation code always receives observations in
[0, 1] exactly as required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from .envs import make_raw_env
from .models import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=["dqn", "a2c"], default="dqn")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--subproc", action="store_true", help="Use subprocess vector envs")
    parser.add_argument("--output", default=None)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_envs = args.n_envs or (8 if args.algo == "a2c" else 1)
    output = (
        Path(args.output)
        if args.output
        else PROJECT_ROOT / "models" / f"{args.algo}_trained.zip"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    def make_one(rank: int):
        def _thunk():
            return make_raw_env(seed=args.seed + rank)

        return _thunk

    env_fns = [make_one(i) for i in range(n_envs)]
    vec_cls = SubprocVecEnv if args.subproc else DummyVecEnv
    venv = VecMonitor(vec_cls(env_fns))

    if args.algo == "dqn":
        # Mirrors the RL Zoo Pong DQN settings while keeping a compact replay
        # buffer that fits on a single consumer GPU.
        model = DQN(
            "CnnPolicy",
            venv,
            learning_rate=1e-4,
            buffer_size=10_000,
            learning_starts=100_000,
            batch_size=32,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1_000,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            optimize_memory_usage=False,
            tensorboard_log=None,
            seed=args.seed,
            device=args.device,
            verbose=1,
        )
    else:
        from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

        model = A2C(
            "CnnPolicy",
            venv,
            learning_rate=7e-4,
            n_steps=5,
            ent_coef=0.01,
            vf_coef=0.25,
            policy_kwargs=dict(
                optimizer_class=RMSpropTFLike,
                optimizer_kwargs=dict(eps=1e-5),
            ),
            tensorboard_log=None,
            seed=args.seed,
            device=args.device,
            verbose=1,
        )

    model.learn(total_timesteps=args.timesteps, progress_bar=args.progress)
    model.save(str(output))
    venv.close()
    print(f"Saved {args.algo.upper()} checkpoint to {output}")
    print(
        "This checkpoint stores raw-[0,255] weights. "
        "Use rai.evaluate (which rescales the first conv for [0,1] inputs)."
    )


if __name__ == "__main__":
    main()
