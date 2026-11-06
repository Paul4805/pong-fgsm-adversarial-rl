"""Download the RL Zoo DQN and A2C PongNoFrameskip-v4 checkpoints."""

from __future__ import annotations

import argparse

from .models import download_pretrained


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for algo, path in download_pretrained(force=args.force).items():
        print(f"{algo}: {path}")


if __name__ == "__main__":
    main()
