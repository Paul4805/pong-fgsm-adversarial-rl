# Defense benchmark: FGSM attacks on Atari Pong

**Environment:** `PongNoFrameskip-v4` equivalent (`ALE/Pong-v5`, frame skip 4,
grayscale, 84×84, 4-frame stack, `[0,1]` normalization)  
**Target:** pre-trained RL Zoo DQN  
**Transfer surrogate:** pre-trained RL Zoo A2C  
**Evaluation:** 10 paired seeds per condition, greedy actions  
**Hardware:** NVIDIA RTX 3060 Laptop GPU, 6 GB VRAM

## Executive summary

- **Defense C (vector quantization) is the best practical defense** in this
  benchmark. It preserves clean return exactly (20.70) and reduces all three
  FGSM attacks to 20.50, essentially eliminating the attack-induced reward drop
  (0.20 points) at an extra inference cost of **0.70 ms/step** and **1.13 s** of
  training.
- **Defense D (adversarial fine-tuning) preserves clean performance and improves
  black-box transfer robustness** (16.90 under A2C→DQN transfer versus −4.70
  undefended), but remains vulnerable to an **adaptive white-box** attack
  (−18.30 at ε=0.001 and −20.70 at ε=0.005).
- **Defense A (standalone denoising autoencoder) fails**: it lowers clean
  return from 20.70 to −20.70. Its reconstruction MSE is small, but pixel-level
  similarity is not enough to preserve the DQN's nearly-flat Q-value ordering.
- **Defense B (detect + denoise)** detects most attacked frames (TPR 73.7% at
  white-box ε=0.005, 66.2% at black-box ε=0.005) with a 1.22% clean FPR, but the
  shared autoencoder still fails to recover reward: all attacked returns remain
  between −18.0 and −19.8.

The robustness-vs-cost Pareto frontier is therefore dominated by **C** on
robustness and training cost, with **D** remaining attractive only when
additional inference latency must be zero.

## 1. Protocol

The existing attack protocol is unchanged:

| Condition | Attack | Gradient source | ε |
|---|---|---|---|
| Clean | none | — | — |
| White-box small | FGSM | target DQN | 0.001 |
| White-box large | FGSM | target DQN | 0.005 |
| Black-box | FGSM | A2C surrogate | 0.005 |

For A/B/C, the white-box FGSM gradient is computed through the frozen,
undefended DQN, matching the established benchmark protocol. For D, the
white-box attack is computed through the fine-tuned DQN. The black-box attack
always transfers from A2C.

All evaluations use the same 10 seeds. Reward drop is measured against the
clean undefended baseline (20.70). Action-change rate is the fraction of steps
where the defended policy's action differs from the clean undefended DQN's
action on the same underlying observation.

## 2. Main results

| Defense | Clean | WB ε=0.001 | WB ε=0.005 | BB ε=0.005 | Mean attack drop | Action change | Extra latency | Train time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Undefended DQN | 20.70 | −13.80 | −19.80 | −4.70 | 33.47 | 71.6% | 0.00 ms | 0.0 s |
| A: Denoising AE | −20.70 | −19.80 | −20.60 | −20.80 | 41.10 | 68.2% | 1.26 ms | 50.6 s |
| B: Detect + denoise | 20.70 | −18.00 | −19.80 | −18.70 | 39.53 | 71.2% | 1.35 ms | 50.6 s |
| C: Vector quantization | **20.70** | **20.50** | **20.50** | **20.50** | **0.20** | 49.9% | **0.70 ms** | **1.1 s** |
| D: Adversarial fine-tune | **21.00** | −18.30 | −20.70 | **16.90** | 28.07 | 64.9% | **0.00 ms** | 37.5 s |

Per-condition recovery relative to the undefended attack result:

| Defense | WB ε=0.001 recovery | WB ε=0.005 recovery | BB ε=0.005 recovery |
|---|---:|---:|---:|
| A | −17.4% | −2.0% | −63.4% |
| B | −12.2% | 0.0% | −55.1% |
| C | **99.4%** | **99.5%** | **99.2%** |
| D | −13.0% | −2.2% | **85.0%** |

## 3. Defense-by-defense analysis

### A — Denoising autoencoder (DQWAE-style, simplified)

**Implementation.** A standalone convolutional autoencoder with a 32×11×11
latent map (133,777 parameters) is trained on 18,035 clean Pong frames. The
training target is always the clean frame; inputs are mixed between clean
frames, Gaussian noise (σ=0.01), and FGSM-like sign noise (ε=0.005). At
evaluation every observation is reconstructed before the frozen DQN.

**Empirical performance.** Clean return collapses to **−20.70**. The
reconstruction MSE is low on both clean frames (**1.10×10⁻⁴**) and
FGSM-perturbed frames (**1.22×10⁻⁴**), but this pixel-level metric is
misleading: the clean action changes on **63.5%** of steps, and all three
attacked conditions remain near −20.

**Cost.** 133,777 parameters, 50.6 s training, 1.26 ms extra latency per step.

**Why it fails.** The DQN's Q-values are nearly flat; small reconstruction
artifacts are enough to change the argmax. The paper's DQWAE integrates the
encoder with the Q-network and trains the policy and reconstruction objectives
together. This benchmark deliberately tests the standalone modular
simplification, and the result shows that policy-aware training is essential.

**Limitations.** The autoencoder is not optimized for the policy loss,
over-smooths task-relevant edges, and is not evaluated against an adaptive
attack through the autoencoder. A residual blend with the raw observation
restored clean play in exploratory tests but also removed the defense effect.

### B — Detect and denoise

**Implementation.** The same autoencoder supplies a mean per-frame
reconstruction error. The threshold is calibrated to **1% clean FPR** on 5,015
held-out clean observations. Flagged observations are denoised; unflagged
observations go directly to the DQN.

**Detector performance.** Calibration threshold: **1.594×10⁻⁴**.
Calibration FPR: **1.02%**. Evaluation FPR:
**1.22%**. TPR at white-box ε=0.005: **73.7%**; TPR at black-box ε=0.005:
**66.2%**. The detector is doing its job statistically.

**Reward performance.** Clean return is preserved at **20.70**, but attacked
returns are **−18.00**, **−19.80**, and **−18.70**, all worse than the
undefended black-box result. Detection does not translate into recovery because
the denoiser itself changes clean task-relevant features.

**Cost.** Shared autoencoder; 1.35 ms/step including the detector forward pass.

**Sensitivity.** A 5% FPR operating point was also tested. It increased
interventions and reduced clean return to **6.40** with a 19.8% measured clean
flag rate, while attacked returns remained around −19. This is a strong
warning that the detector/denoiser operating point must be tuned jointly with
the policy, not just for detection accuracy.

**Limitations.** The detector is statistical, not certified; threshold drift
under closed-loop control causes cascading false positives. The denoiser is the
same policy-unaware AE as Defense A, so detection alone cannot compensate for
its information loss.

### C — Vector quantization (VQ)

**Implementation.** Each 84×84 frame is split into 6×6 patches. A codebook of
**K=128** patch prototypes is trained with mini-batch k-means on 200,000 clean
patches. At evaluation every patch is replaced by its nearest codebook entry.

**Metrics.** Clean-frame quantization distortion: **3.54×10⁻⁵ MSE**.
Codebook utilization on clean validation patches: **80.5%**. Training time:
**1.13 s**. Extra inference latency: **0.70 ms/step**.

**Empirical performance.** Clean return is **20.70**, identical to the
undefended baseline. All three attacked conditions return **20.50** — a reward
drop of only **0.20** and roughly **99% recovery** relative to the undefended
attack. The action-change rate under attack is 47–52%, but those alternative
actions are equally good: the policy still wins 20–21 points per episode.

**Interpretation.** FGSM perturbations are high-frequency, small-amplitude
changes. Projecting each patch onto a clean prototype removes that perturbation
while preserving the coarse game structure (ball, paddles, score). The codebook
is extremely cheap and becomes the dominant factor, so the same result is
obtained for white-box and black-box transfer.

**Limitations.** Nearest-neighbor VQ is non-differentiable, so true adaptive
white-box attacks require a straight-through or gradient-free approximation.
A codebook trained on Pong may not transfer to visually richer games, and a
very small K would erase the ball. Within this benchmark, however, the
distortion/utilization trade-off is favorable.

### D — Adversarial fine-tuning (RS-DQN-style, lightweight)

**Implementation.** The pre-trained DQN is fine-tuned offline for at most 8
epochs (early stopping after two non-improving clean evaluations) on 6,000
clean transitions. Each batch mixes clean states with on-the-fly FGSM states
generated from the current Q-network. The final objective is a **margin-1.0**
adversarial loss on both clean and perturbed states; training uses 75% ε=0.01
and 25% ε=0.005 and stops after 4 effective epochs.

**Empirical performance.** Clean return is **21.00** — no clean-performance
trade-off. Under black-box transfer it reaches **16.90**, recovering **85%** of
the clean return and substantially beating the undefended black-box result
(−4.70). Under adaptive white-box attack, however, it remains vulnerable:
**−18.30** at ε=0.001 and **−20.70** at ε=0.005.

**Cost.** 37.5 s training, 0 ms additional inference latency, peak VRAM
**172 MB**. The fine-tuned model has the same 3.37M parameters as the baseline.

**Interpretation.** The margin objective successfully preserves the clean
policy and improves transfer robustness, but an adaptive attacker that
differentiates through the fine-tuned Q-network can still find small
perturbations that flip the nearly-flat Q-values. This is the expected gap
between a lightweight offline approximation and the paper's online RS-DQN
student/Q-network training. We tested three variants: standard TD
fine-tuning collapsed clean return to −2.0; teacher-Q distillation preserved
clean return but did not improve white-box robustness; the margin objective was
the best clean-preserving variant and is the one reported here.

**Limitations.** Only 6,000 transitions and a few epochs; no online environment
sampling or student/Q split; the training objective uses FGSM perturbations, so
it can overfit the specific budgets. The black-box improvement should not be
mistaken for adaptive white-box robustness.

## 4. Robustness vs. cost Pareto comparison

| Defense | Clean drop | Mean attack drop | Extra latency | Training time | Parameters |
|---|---:|---:|---:|---:|---:|
| A | 41.40 | 41.10 | 1.26 ms | 50.6 s | 133,777 |
| B | 0.00 | 39.53 | 1.35 ms | 50.6 s | 133,777 |
| C | **0.00** | **0.20** | **0.70 ms** | **1.1 s** | 4,608 |
| D | **−0.30** | 28.07 | **0.00 ms** | 37.5 s | 3,374,412 |

**Conclusion.** C is the clear practical winner: it preserves clean play,
removes nearly all of the attack reward drop, trains in about one second, and
adds only 0.70 ms/step. D has lower inference cost and better black-box transfer
robustness, but it costs far more training and fails under adaptive white-box
attack. A and B are dominated on robustness and clean performance.

## 5. Compute time

Measured on the RTX 3060 Laptop:

| Stage | Time |
|---|---:|
| Clean data collection (12 episodes) | 65.9 s |
| Autoencoder training (12 epochs) | 50.6 s |
| VQ k-means training | 1.13 s |
| Adversarial fine-tuning (4 effective epochs) | 37.5 s |
| Evaluation (5 defenses × 4 conditions × 10 episodes) | 1150.8 s |
| **Total** | **≈ 21.8 min** |

## 6. Reproduction

```bash
python -m rai.benchmark --stage all
```

The run writes:

```
results/defense_benchmark/summary.csv
results/defense_benchmark/summary_long.csv
results/defense_benchmark/episodes.csv
results/defense_benchmark/benchmark_metadata.json
results/defense_benchmark/plots/
results/defense_benchmark/frames/
```

Auxiliary checkpoints are written to `artifacts/defense_benchmark/` and are
excluded from Git. Use `--stage train` and `--stage evaluate` to split the
pipeline.

## 7. Limitations

- White-box attacks for A/B/C are generated through the frozen undefended DQN,
  matching the established benchmark rather than adapting through the defense.
  Adaptive attacks through the autoencoder or VQ could be stronger.
- D's white-box result is adaptive because the attack differentiates through
  the fine-tuned DQN.
- The benchmark uses deterministic actions, 10 episodes per condition, and the
  visually simple game of Pong.
- A/B/D are deliberately lightweight approximations of the published methods;
  they are not full reproductions of DQWAE, the ECML/PKDD detector/denoiser, or
  online RS-DQN.

## References

1. Y. Ohashi et al., “Deep Adversarial Reinforcement Learning With Noise
   Compensation by Autoencoder,” *IEEE Access*, vol. 9, pp. 143901–143912,
   2021. DOI: [10.1109/ACCESS.2021.3121751](https://doi.org/10.1109/ACCESS.2021.3121751)
2. X. Xiong et al., “Defending Observation Attacks in Deep Reinforcement
   Learning via Detection and Denoising,” *ECML/PKDD 2022*, pp. 242–257.
   DOI: [10.1007/978-3-031-26409-2_15](https://doi.org/10.1007/978-3-031-26409-2_15)
3. Luu et al., “Mitigating Adversarial Perturbations for Deep Reinforcement
   Learning via Vector Quantization,” *IROS 2024*, pp. 595–602.
   [arXiv:2410.03376](https://arxiv.org/abs/2410.03376)
4. Fischer et al., “Online Robustness Training for Deep Reinforcement
   Learning,” 2019. [arXiv:1911.00887](https://arxiv.org/abs/1911.00887)
