"""7.5  Noise robustness, combined: input noise + weight noise + label smoothing together.

The three components live in their own modules (input_noise, weight_noise, label_smoothing);
this method composes them and tunes the three strengths jointly."""
from .base import Method, soft_cross_entropy
from .input_noise import add_input_noise
from .weight_noise import perturbed_weights


class NoiseRobustness(Method):
    name = "noise_robustness"
    title = "Noise robustness (inputs + weights + targets)"
    section = "7.5"

    def defaults(self):
        return {"lr": 0.05, "sigma_in": 0.05, "sigma_w": 0.005, "eps": 0.05}

    def space(self, trial):
        return {"sigma_in": trial.suggest_float("sigma_in", 0.01, 0.4, log=True),
                "sigma_w": trial.suggest_float("sigma_w", 1e-3, 3e-2, log=True),
                "eps": trial.suggest_float("eps", 0.01, 0.2)}

    def transform(self, batch, model, hp):
        return {**batch, "x": add_input_noise(batch["x"], hp["sigma_in"])}

    def forward_context(self, model, hp):
        return perturbed_weights(model, hp["sigma_w"])

    def data_loss(self, model, batch, hp):
        return soft_cross_entropy(model(batch["x"]), batch["y"], hp["eps"])
