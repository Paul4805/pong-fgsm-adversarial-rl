# Model checkpoints

Pretrained checkpoints are intentionally **not** committed to Git. They are
large binaries and can be reproduced with the Stable-Baselines3 RL Zoo.

Download them with:

```bash
python -m rai.download_models
```

Expected files:

```
models/
├── dqn/
│   └── model.zip    # sb3/dqn-PongNoFrameskip-v4, target policy
└── a2c/
    └── model.zip    # sb3/a2c-PongNoFrameskip-v4, black-box surrogate
```

The first time `rai.evaluate` loads the DQN checkpoint it creates
`models/dqn/model_compat.zip`, a compatibility-patched copy for newer
Stable-Baselines3 versions. The policy weights are unchanged.
