"""Fast Gradient Sign Method for Q-value and actor-critic policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .models import PolicyAgent


@dataclass
class AttackOutput:
    adv_obs: np.ndarray
    delta: np.ndarray
    loss: float
    gradient_action: int
    label_action: int
    grad_abs_mean: float
    max_abs_delta: float
    mean_abs_delta: float


class FGSMAttacker:
    """Untargeted L_inf FGSM.

    The attack maximizes a cross-entropy policy loss with respect to the input
    observation.  For DQN the logits are Q-values and the label is the greedy
    action; for A2C the logits are the categorical policy logits.  Maximizing
    the loss pushes the policy away from its current action.

    ``adv = clip(obs + epsilon * sign(grad_obs(loss)), 0, 1)``
    """

    def __init__(
        self,
        gradient_agent: PolicyAgent,
        epsilon: float,
        label_source: str = "own",
    ) -> None:
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        if label_source not in {"own", "target"}:
            raise ValueError("label_source must be 'own' or 'target'")
        self.gradient_agent = gradient_agent
        self.epsilon = float(epsilon)
        self.label_source = label_source

    def perturb(
        self,
        obs: np.ndarray,
        target_clean_action: int | None = None,
    ) -> AttackOutput:
        """Compute an FGSM perturbation for one stacked observation.

        Parameters
        ----------
        obs:
            Clean ``(4, 84, 84)`` float32 observation in [0, 1].
        target_clean_action:
            If ``label_source='target'`` this is the label to maximize the loss
            against.  For white-box attacks it is the same as the gradient
            agent's action.
        """
        clean = np.asarray(obs, dtype=np.float32)
        obs_t = self.gradient_agent.tensor(clean, requires_grad=True)
        logits = self.gradient_agent.forward_logits(obs_t)
        own_action = int(logits.argmax(dim=1).item())

        if self.label_source == "target":
            if target_clean_action is None:
                raise ValueError("target_clean_action is required when label_source='target'")
            label = int(target_clean_action)
        else:
            label = own_action

        labels = torch.tensor([label], device=logits.device, dtype=torch.long)
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, obs_t, retain_graph=False, create_graph=False)[0]

        if self.epsilon == 0.0:
            adv_t = obs_t.detach()
        else:
            adv_t = (obs_t.detach() + self.epsilon * torch.sign(grad)).clamp_(0.0, 1.0)

        adv = adv_t[0].detach().cpu().numpy().astype(np.float32)
        delta = adv - clean
        return AttackOutput(
            adv_obs=adv,
            delta=delta,
            loss=float(loss.detach().item()),
            gradient_action=own_action,
            label_action=label,
            grad_abs_mean=float(grad.abs().mean().item()),
            max_abs_delta=float(np.abs(delta).max()),
            mean_abs_delta=float(np.abs(delta).mean()),
        )
