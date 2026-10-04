"""7.13  Adversarial training (fast gradient sign method).

x_adv = clip(x + eps * sign(grad_x CE(x, y)), 0, 1);  loss = a*CE(x) + (1-a)*CE(x_adv).
Encourages local constancy of the learned function around the training points.  Robust (FGSM)
accuracy is reported for every method, so the effect is visible against the baseline."""
import torch
import torch.nn.functional as F

from .base import Method


class AdversarialTraining(Method):
    name = "adversarial"
    title = "Adversarial training (FGSM)"
    section = "7.13"

    def defaults(self):
        return {"lr": 0.05, "eps": 0.1, "mix": 0.5}

    def space(self, trial):
        return {"eps": trial.suggest_float("eps", 0.01, 0.3),
                "mix": trial.suggest_float("mix", 0.3, 0.9)}

    def data_loss(self, model, batch, hp):
        x = batch["x"].clone().requires_grad_(True)
        clean = F.cross_entropy(model(x), batch["y"])
        (g,) = torch.autograd.grad(clean, x, retain_graph=True)
        x_adv = (x + hp["eps"] * g.sign()).clamp(0, 1).detach()
        adv = F.cross_entropy(model(x_adv), batch["y"])
        return hp["mix"] * clean + (1 - hp["mix"]) * adv
