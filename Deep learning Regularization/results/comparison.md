| # | Method | Book § | Regime | Val acc | Test acc ± SE | Test AUC | Test loss | FGSM acc (val) | Tuned hyperparameters |
|---|---|---|---|---|---|---|---|---|---|
| 0 | Baseline (no regularization) | reference | full labels | 0.9801 | 0.9797 ± 0.0014 | 0.99972 | 0.0886 | 0.483 | lr=0.05 |
| 1 | L2 regularization (weight decay) | 7.1.1 | full labels | 0.9802 | 0.9801 ± 0.0014 | 0.99976 | 0.0673 | 0.242 | lr=0.0242, alpha=5.63e-05 |
| 2 | L1 regularization | 7.1.2 | full labels | 0.9806 | 0.9799 ± 0.0014 | 0.99973 | 0.0724 | 0.391 | lr=0.05, alpha=1e-05 |
| 3 | Norm penalties as constrained optimization (max-norm) | 7.2 | full labels | 0.9793 | 0.9786 ± 0.0014 | 0.99970 | 0.0770 | 0.238 | lr=0.0203, max_norm=2.98 |
| 4 | Regularization and under-constrained problems | 7.3 | 500 training examples (< 784 inputs) | 0.8718 | 0.8791 ± 0.0033 | 0.98887 | 0.5043 | 0.103 | lr=0.05, alpha=0.001 |
| 5 | Dataset augmentation | 7.4 | full labels | 0.9859 | 0.9874 ± 0.0011 | 0.99988 | 0.0393 | 0.110 | lr=0.0197, max_shift=1.63, max_rot=13.7, max_scale=0.00453 |
| 6 | Noise robustness (inputs + weights + targets) | 7.5 | full labels | 0.9818 | 0.9811 ± 0.0014 | 0.99924 | 0.1490 | 0.423 | lr=0.0837, sigma_in=0.0167, sigma_w=0.0027, eps=0.0796 |
| 7 | Noise on inputs | 7.5 (inputs) | full labels | 0.9830 | 0.9830 ± 0.0013 | 0.99975 | 0.0798 | 0.539 | lr=0.05, sigma=0.1 |
| 8 | Noise on weights | 7.5 (weights) | full labels | 0.9825 | 0.9802 ± 0.0014 | 0.99974 | 0.0785 | 0.354 | lr=0.0316, sigma=0.00588 |
| 9 | Noise on output targets (label smoothing) | 7.5.1 | full labels | 0.9818 | 0.9812 ± 0.0014 | 0.99948 | 0.1881 | 0.306 | lr=0.0593, eps=0.114 |
| 10 | Semi-supervised learning (shared-encoder autoencoder) | 7.6 | 1000 labels + unlabeled rest | 0.8910 | 0.8991 ± 0.0030 | 0.99251 | 0.4219 | 0.049 | lr=0.0167, lam=0.309 |
| 11 | Multitask learning (digit + parity + magnitude) | 7.7 | full labels | 0.9803 | 0.9835 ± 0.0013 | 0.99974 | 0.0722 | 0.295 | lr=0.0242, lam=0.119 |
| 12 | Early stopping | 7.8 | full labels | 0.9795 | 0.9781 ± 0.0015 | 0.99967 | 0.0788 | 0.328 | lr=0.05, patience=5 |
| 13 | Parameter tying and parameter sharing | 7.9 | full labels | 0.9799 | 0.9816 ± 0.0013 | 0.99967 | 0.1155 | 0.682 | lr=0.05, mode=hard, depth=3, lam_tie=0.1 |
| 14 | Sparse representations (activation L1) | 7.10 | full labels | 0.9825 | 0.9814 ± 0.0014 | 0.99967 | 0.0713 | 0.296 | lr=0.05, lam=0.01 |
| 15 | Bagging (ensemble of bootstrap-trained MLPs) | 7.11 | ensemble of 5 MLPs | 0.9822 | 0.9832 ± 0.0013 | 0.99978 | 0.0612 | 0.730 | lr=0.05 |
| 16 | Dropout | 7.12 | full labels | 0.9839 | 0.9818 ± 0.0013 | 0.99975 | 0.0612 | 0.615 | lr=0.0408, p_in=0.236, p_hidden=0.14 |
| 17 | Adversarial training (FGSM) | 7.13 | full labels | 0.9839 | 0.9858 ± 0.0012 | 0.99983 | 0.0569 | 0.837 | lr=0.0788, eps=0.0552, mix=0.394 |
| 18 | Tangent prop / tangent distance / manifold tangents | 7.14 | full labels | 0.9802 | 0.9806 ± 0.0014 | 0.99977 | 0.0702 | 0.312 | lr=0.0281, lam=2.02, source=affine |
