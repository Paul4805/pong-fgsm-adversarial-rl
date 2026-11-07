# Changelog

## 0.2.0 — 2026-11-07

- Added an end-to-end defense benchmark for FGSM attacks on the pre-trained
  Pong DQN.
- Added four defense implementations:
  - denoising convolutional autoencoder,
  - reconstruction-error detect-and-denoise pipeline,
  - patch-wise vector quantization,
  - lightweight offline adversarial fine-tuning.
- Added paired clean/white-box/black-box evaluation over 10 episodes per
  condition.
- Added defense plots, frame comparisons, summary CSV files, a benchmark
  metadata file, and a critical analysis report.
- Added unit tests for the defense components and updated CI.

## 0.1.0 — 2026-11-06

- Initial release: white-box and black-box FGSM against a DQN Pong policy,
  with A2C transfer surrogate, paired evaluation, result files, and
  visualizations.
