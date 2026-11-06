# FGSM Pong attack results

- Environment: `PongNoFrameskip-v4` (`ALE/Pong-v5` + frame skip 4)
- Target: **DQN**
- Black-box surrogate: **A2C**
- Evaluation episodes per condition: 10
- Device: `cuda`

Action-change rate is the fraction of attacked steps where the target policy's action on the perturbed observation differs from its action on the same clean observation. The parenthesized value uses all evaluation steps as the denominator; the first value uses attacked steps only.

| Mode | epsilon | Clean return | Adversarial return | Reward drop | Action change |
|---|---:|---:|---:|---:|---:|
| whitebox | 0.005 | 20.70 | -19.80 | +40.50 | 89.10% (89.10%) |
| whitebox | 0.001 | 20.70 | -13.80 | +34.50 | 81.88% (81.88%) |
| blackbox | 0.005 | 20.70 | -4.70 | +25.40 | 58.89% (58.89%) |
| blackbox | 0.001 | 20.70 | 21.00 | -0.30 | 18.13% (18.13%) |

## Clean baseline

Mean return over 10 episodes: **20.70** (std 0.46).

## Attack definition

For every attacked step the attacker computes

```
loss = CrossEntropy(policy_logits(s), a_clean)
s_adv = clip(s + epsilon * sign(grad_s loss), 0, 1)
```

and acts with the target policy on `s_adv`. For DQN the logits are Q-values; for A2C they are categorical policy logits. The white-box attack differentiates the target; the black-box attack differentiates the A2C surrogate and transfers `s_adv` to the DQN target.
