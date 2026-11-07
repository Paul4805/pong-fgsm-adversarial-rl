"""Lightweight offline adversarial fine-tuning of the pre-trained DQN (Defense D)."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..models import PolicyAgent


def _batch_from_u8(obs: np.ndarray, idx: torch.Tensor, device: torch.device) -> torch.Tensor:
    # Index a uint8 CPU array on CPU first, then transfer the small batch.
    idx_np = idx.detach().cpu().numpy()
    batch = np.asarray(obs[idx_np], dtype=np.float32) / 255.0
    return torch.as_tensor(batch, device=device, dtype=torch.float32)


def _fgsm_batch(
    q_net: torch.nn.Module,
    obs: torch.Tensor,
    actions: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    obs_t = obs.detach().clone().requires_grad_(True)
    logits = q_net(obs_t)
    loss = F.cross_entropy(logits, actions)
    grad = torch.autograd.grad(loss, obs_t, retain_graph=False, create_graph=False)[0]
    adv = (obs_t.detach() + epsilon * torch.sign(grad)).clamp_(0.0, 1.0)
    return adv


@dataclass
class RobustFineTuneResult:
    model: Any
    metadata: dict[str, Any]


def fine_tune_dqn(
    agent: PolicyAgent,
    transitions: Any,
    epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    adversarial_fraction: float = 0.5,
    epsilon_choices: Sequence[float] = (0.001, 0.005),
    gamma: float = 0.99,
    seed: int = 0,
    loss_mode: str = "distill",
    teacher_agent: PolicyAgent | None = None,
    distill_weight: float = 1.0,
    margin_weight: float = 0.25,
    margin: float = 0.02,
    clean_eval_fn: Callable[[PolicyAgent, int], float] | None = None,
    clean_eval_episodes: int = 3,
    verbose: bool = True,
) -> RobustFineTuneResult:
    """Fine-tune the DQN Q-network on a mix of clean and FGSM states.

    Two objectives are available:

    * ``loss_mode="td"``: standard one-step TD loss against a frozen target
      network.
    * ``loss_mode="distill"``: robust distillation from the frozen baseline
      Q-network.  Both clean and perturbed observations must reproduce the
      teacher's Q-values, with an additional action-margin term that explicitly
      preserves the clean greedy action on perturbed states.

    The distillation objective is the recommended lightweight approximation of
    robust policy fine-tuning because the baseline Q-values are very flat for
    Pong; unconstrained TD updates can flip the greedy action and collapse the
    clean return.
    """
    if agent.algo != "dqn":
        raise ValueError("Defense D requires a DQN agent")
    if loss_mode not in {"td", "distill", "margin"}:
        raise ValueError("loss_mode must be 'td', 'distill', or 'margin'")
    if loss_mode == "distill" and teacher_agent is None:
        raise ValueError("teacher_agent is required for loss_mode='distill'")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model = agent.model
    q_net = model.policy.q_net
    target_net = model.policy.q_net_target
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    for param in target_net.parameters():
        param.requires_grad_(False)

    teacher_q_net = None
    if teacher_agent is not None:
        teacher_q_net = teacher_agent.model.policy.q_net
        teacher_q_net.eval()
        for param in teacher_q_net.parameters():
            param.requires_grad_(False)

    optimizer = torch.optim.Adam(q_net.parameters(), lr=learning_rate)
    n = len(transitions.actions)
    if n == 0:
        raise ValueError("No transitions available for fine-tuning")

    if torch.cuda.is_available() and agent.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(agent.device)

    def _margin_loss(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        correct = logits.gather(1, actions.unsqueeze(1)).squeeze(1)
        mask = F.one_hot(actions, num_classes=logits.shape[1]).bool()
        other = logits.masked_fill(mask, -1e9).max(dim=1).values
        return F.relu(margin - (correct - other)).mean()

    start_time = time.perf_counter()
    epoch_history: list[dict[str, Any]] = []
    best_state = copy.deepcopy({k: v.detach().cpu() for k, v in q_net.state_dict().items()})
    best_clean_return = -float("inf")
    stale_epochs = 0
    global_step = 0

    for epoch in range(epochs):
        q_net.train()
        order = rng.permutation(n)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, n, batch_size):
            idx_np = order[start : start + batch_size]
            idx = torch.as_tensor(idx_np, device=agent.device)
            clean_obs = _batch_from_u8(transitions.transition_observations, idx, agent.device)
            next_obs = _batch_from_u8(transitions.next_observations, idx, agent.device)
            actions = torch.as_tensor(transitions.actions[idx_np], device=agent.device, dtype=torch.long)
            rewards = torch.as_tensor(transitions.rewards[idx_np], device=agent.device)
            dones = torch.as_tensor(transitions.dones[idx_np], device=agent.device, dtype=torch.float32)

            # Train mostly at the larger budget so the policy sees perturbations
            # at least as strong as the evaluation FGSM budget.
            if len(epsilon_choices) == 1:
                epsilon = float(epsilon_choices[0])
            else:
                epsilon = float(epsilon_choices[-1] if rng.random() < 0.75 else epsilon_choices[0])
            adv_mask = torch.rand(len(clean_obs), device=agent.device) < adversarial_fraction
            if bool(adv_mask.any()):
                with torch.enable_grad():
                    adv_obs = _fgsm_batch(q_net, clean_obs, actions, epsilon)
                mixed_obs = torch.where(adv_mask[:, None, None, None], adv_obs, clean_obs)
            else:
                adv_obs = clean_obs
                mixed_obs = clean_obs

            if loss_mode == "td":
                q_values = q_net(mixed_obs).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_values = target_net(next_obs).max(dim=1).values
                    td_target = rewards + gamma * (1.0 - dones) * next_values
                loss = F.smooth_l1_loss(q_values, td_target)
            elif loss_mode == "distill":
                with torch.no_grad():
                    teacher_q = teacher_q_net(clean_obs)
                clean_q = q_net(clean_obs)
                adv_q = q_net(adv_obs)
                distill = F.mse_loss(clean_q, teacher_q) + F.mse_loss(adv_q, teacher_q)
                ranking = _margin_loss(clean_q, actions) + _margin_loss(adv_q, actions)
                loss = distill_weight * distill + margin_weight * ranking
            else:
                # Margin-only adversarial fine-tuning: preserve the clean action
                # with a large Q-value margin on both clean and FGSM states.
                clean_q = q_net(clean_obs)
                adv_q = q_net(adv_obs)
                loss = margin_weight * (
                    _margin_loss(clean_q, actions) + _margin_loss(adv_q, actions)
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            batches += 1
            global_step += 1

        q_net.eval()
        clean_return = float("nan")
        if clean_eval_fn is not None:
            clean_return = float(clean_eval_fn(agent, clean_eval_episodes))
            if clean_return > best_clean_return:
                best_clean_return = clean_return
                best_state = copy.deepcopy(
                    {k: v.detach().cpu() for k, v in q_net.state_dict().items()}
                )
                stale_epochs = 0
            else:
                stale_epochs += 1
        record = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss / max(batches, 1),
            "clean_return": clean_return,
            "batches": batches,
        }
        epoch_history.append(record)
        if verbose:
            print(
                f"  DQN epoch {epoch + 1:02d}/{epochs}: loss={record['train_loss']:.6f} "
                f"clean_return={clean_return:.2f}"
            )
        # Early stop if the clean policy has not improved for two epochs.
        if clean_eval_fn is not None and stale_epochs >= 2:
            if verbose:
                print("  DQN early stop: clean return has not improved for two epochs")
            break

    if clean_eval_fn is not None and np.isfinite(best_clean_return):
        q_net.load_state_dict({k: v.to(agent.device) for k, v in best_state.items()})
    q_net.eval()
    train_seconds = time.perf_counter() - start_time
    max_vram_mb = None
    if torch.cuda.is_available() and agent.device.type == "cuda":
        max_vram_mb = float(torch.cuda.max_memory_allocated(agent.device) / 1e6)

    metadata = {
        "mode": f"offline_fgsm_adversarial_finetune_{loss_mode}",
        "loss_mode": loss_mode,
        "distill_weight": distill_weight,
        "margin_weight": margin_weight,
        "margin": margin,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "adversarial_fraction": adversarial_fraction,
        "epsilon_choices": list(epsilon_choices),
        "gamma": gamma,
        "transitions": int(n),
        "global_steps": int(global_step),
        "train_seconds": train_seconds,
        "max_vram_mb": max_vram_mb,
        "best_clean_return": best_clean_return if np.isfinite(best_clean_return) else None,
        "history": epoch_history,
    }
    return RobustFineTuneResult(model=model, metadata=metadata)

