# MNIST Regularization Study (Deep Learning book, Chapter 7)

## Project Overview
This repository serves as a clean, modular experimentation framework built to empirically study Deep Learning regularization techniques—as formalized in Chapter 7 of the Deep Learning textbook by Ian Goodfellow et al.—using the MNIST dataset. 

The architecture employs a highly decoupled pattern:
- A central **orchestrator** manages hyperparameter tuning (via Optuna) and execution.
- The **trainer** encapsulates the core Stochastic Gradient Descent (SGD) loop.
- **Tracking** and lifecycle are managed through MLflow.
- Individual **regularization methods** (19 in total) subclass a common base, implementing specific techniques by selectively overriding loss functions, parameter penalties, or data transformations. 
- An embedded `Normalize` layer inside the `MLP` prevents train/serve skew, and strict data separation enforces a 50k/10k/10k train/val/test split to guarantee experimental integrity.

---

## Results Analysis & Conclusion

The full empirical outputs can be found in the `results/` directory, summarized in `results/comparison.md`. 

**Baseline**: The unregularized baseline model achieves a test accuracy of ~**97.97%** with an FGSM robustness of **48.3%**.

**Best Overall Methods**:
1. **Dataset Augmentation (`augmentation`)**: Achieved the highest overall test accuracy at **~98.74%**, conclusively proving that algorithmically increasing the volume and variance of training data is often the most effective form of regularization.
2. **Adversarial Training (`adversarial`)**: While yielding a highly competitive test accuracy of **~98.58%**, its standout feature is robustness. It drastically improved the FGSM (Fast Gradient Sign Method, $\epsilon = 0.1$) validation accuracy to **83.7%** (up from the baseline's 48.3%).

*Conclusion*: For raw generalization performance on this dataset, Data Augmentation is the reigning champion. However, if robustness to slight input perturbations or adversarial attacks is a concern, Adversarial Training provides an unparalleled defense with minimal hit to generalization. Methods designed for low-data regimes (`under_constrained`, `semi_supervised`) behave as mathematically expected but are segregated from the primary accuracy leaderboard.

---

## File Explanations & Architecture

* **`run.py`**: The main entry point to execute the pipeline. Triggers the orchestrator for all methods, specific methods, or fast/smoke tests.
* **`infer.py`**: The inference system. Runs predictions using the bundled/registered models.
* **`regpipe/orchestrator.py`**: Drives the evaluation pipeline, coordinating Optuna tuning, initiating the final training runs, handling artifact creation, and model registration.
* **`regpipe/trainer.py`**: Contains the generic PyTorch SGD training loop. It delegates variations (like step adjustments or custom losses) to the individual method subclasses.
* **`regpipe/model.py`**: Defines the core network architecture. Crucially, it embeds data standardization (`Normalize`) within the PyTorch model graph itself to prevent data skew during inference.
* **`regpipe/data.py`**: Handles deterministic data loading. It reliably carves out a 10K validation set from the MNIST training data and rigorously audits the splits for size, class balance, and duplicates.
* **`regpipe/methods/*.py`**: 19 discrete modules, each implementing a specific Chapter 7 regularization algorithm.

---

## The 19 Methods (`regpipe/methods/`)
Each file implements a method by changing how the model learns. From an ML perspective:

| Module | Book § | ML/Regularization Mechanism | Tuned |
|---|---|---|---|
| `baseline` | – | Nothing (Reference). Standard empirical risk minimization. | lr |
| `l2_weight_decay` | 7.1.1 | **Weight Decay**: Drives weights towards the origin, reducing model complexity and limiting the influence of any single feature. | α |
| `l1` | 7.1.2 | **Sparsity**: L1 penalty forces many weights to exactly zero, essentially performing automatic feature selection. | α |
| `constrained_norm` | 7.2 | **Max-Norm**: Hard constraint on the weights' norm (reprojected after steps), preventing explosive weights often seen in deep networks. | c |
| `under_constrained` | 7.3 | **Well-Posedness**: Uses 500 examples (< 784 inputs) making the problem under-constrained. Applies L2 to make the matrix invertible/well-posed. | α |
| `augmentation` | 7.4 | **Data Augmentation**: Applies random affine transforms (shift/rot/zoom). Simulates learning invariance to spatial transformations. | shift, rot, scale |
| `noise_robustness` | 7.5 | **Combined Noise**: Combines inputs, weights, and targets noise for broad robustness. | σ_in, σ_w, ε |
| `input_noise` | 7.5 | **Input Noise**: Injects Gaussian noise to inputs, making the model insensitive to tiny, high-frequency data changes. | σ |
| `weight_noise` | 7.5 | **Weight Noise**: Evaluates gradient at noisy weights. Pushes the optimizer to find wide, flat, robust minima rather than sharp valleys. | σ |
| `label_smoothing` | 7.5.1 | **Target Smoothing**: Replaces hard 1s and 0s with 0.9 and 0.1/K. Stops the model from becoming overly confident and pushing weights to infinity. | ε |
| `semi_supervised` | 7.6 | **Semi-Supervised**: Uses an autoencoder on unlabeled data to learn a shared representation, aiding the classifier trained on a small labeled subset. | λ |
| `multitask` | 7.7 | **Multitask Learning**: Forces the shared hidden layers to also predict digit parity and magnitude. Acts as a regularizer by forcing generalizable features. | λ |
| `early_stopping` | 7.8 | **Early Stopping**: Halts training when validation loss stops improving. Acts as implicit capacity control (limiting the number of optimization steps). | patience |
| `parameter_sharing` | 7.9 | **Tying/Sharing**: Reuses weights across depth or penalizes distance between layers. Drastically cuts the parameter count or constrains parameter space. | mode, depth, λ |
| `sparse_representations` | 7.10 | **Activation Sparsity**: Penalizes non-zero hidden unit activations (L1). Forces the network to represent concepts with as few active neurons as possible. | λ |
| `bagging` | 7.11 | **Ensemble Method**: Averages predictions of 5 independently trained models. Reduces variance without changing bias. | lr |
| `dropout` | 7.12 | **Dropout**: Randomly zeroes out units. Prevents complex co-adaptations; mathematically equivalent to averaging an exponential number of sub-networks. | p_in, p_hidden |
| `adversarial` | 7.13 | **Adversarial Training**: Mixes FGSM-generated adversarial examples into the training loss, explicitly forcing the decision boundary to resist worst-case perturbations. | ε, mix |
| `tangent` | 7.14 | **Tangent Prop**: Forces the model's output to be invariant to known, small transformations (tangents) of the input data manifold. | λ, source |

---

## Data and Splits
MNIST **is** pre-split by its authors into 60,000 train / 10,000 test images, with **no validation set**.
The pipeline carves a stratified 10,000-image validation set out of the training set
(50,000 / 10,000 / 10,000), audits sizes, class balance and exact duplicate images across splits, and
logs the data hash. If torchvision cannot download MNIST (hosts do get blocked), use
`--set data.source=npz:/path/mnist.npz` (keys `x_train,y_train,x_test,y_test`).

## Protocol Decisions You Should Know About
* **Fixed epoch budget.** Every method trains the same number of epochs and is evaluated on its **last**
  epoch's weights, so differences come from the regularizer. `early_stopping` is the only method that
  selects a (best-validation) epoch.
* **Tuning is part of `run.py`.** One Optuna study per method (default configuration is always trial 0),
  scored on validation accuracy, every trial a nested MLflow run. The test set is never read during tuning.
* **Test set is evaluated once per method**. With 19 methods that is 19
  test evaluations, so the "best" method is chosen on **validation** accuracy, never test.
* **Reduced-data regimes are labelled and excluded from champion selection:** `under_constrained`
  (500 training examples) and `semi_supervised` (1,000 labels). Their numbers are not comparable with the rest.
* **Manifold tangent classifier.** This implementation uses local-PCA tangents from same-class neighbours, same idea (data-driven tangents), not
  the same algorithm. Tangent prop with analytic affine tangents is the exact method.
* **Single seed.** The reported SE is the test-set sampling error, not seed-to-seed variance. Run several seeds (`--set seed=N`) to confirm tiny margins.

## Execution and Options
```bash
python run.py                       # all 19 methods on real MNIST (torchvision downloads it)
python run.py --fast                # 2 epochs, 2 trials per method
python run.py --smoke               # synthetic MNIST-shaped data: checks the plumbing
python run.py --methods l2_weight_decay,dropout,adversarial
mlflow ui --backend-store-uri sqlite:///results/mlflow.db
python -m pytest -q                 # 72 tests
```

## MLflow and Registry
Runs are nested: orchestrator > method > trials. Params, per-epoch metrics, final val/test metrics, plots and
bundles are logged. The best **full-data** method (by validation accuracy) is logged as a pyfunc model with a
signature and input example, registered as `MNIST-Reg` with alias `candidate`, and reloaded in a **fresh
process** to confirm predictions. `champion` is set only with `--promote --approver "Name"`.

## Inference
```bash
python infer.py --model results/methods/dropout/bundle --image digit.png
python infer.py --model results/methods/bagging/bundle --npy batch.npy
python infer.py --model results/methods/dropout/bundle --test-index 0 1 2
python infer.py --model "models:/MNIST-Reg@candidate" --tracking-uri sqlite:///results/mlflow.db --image d.png
mlflow models serve -m "models:/MNIST-Reg@candidate" --env-manager local
```
`regpipe/bundle.py` (`Predictor`) needs only `bundle.json` + `model.pt`; no training code, config or data.
