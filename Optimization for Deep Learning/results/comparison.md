| # | Group | Method | Book § | Regime | Data passes | Val acc | Test acc ± SE | Test AUC | Test loss | Tuned hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | reference | Baseline: SGD + momentum + exponential decay, He init | reference | full labels | 20 | 0.9807 | 0.9823 ± 0.0013 | 0.99977 | 0.0558 | lr=0.0408, l2=0.000514, l1=9.96e-09 |
| 1 | update rules | Batch (full-dataset) gradient descent | 8.1.3 | full labels | 100 | 0.9145 | 0.9238 ± 0.0027 | 0.99474 | 0.2669 | lr=0.0653, l2=0.00157, l1=1.01e-06 |
| 2 | update rules | Minibatch stochastic gradient descent | 8.3.1 | full labels | 20 | 0.9789 | 0.9803 ± 0.0014 | 0.99969 | 0.0833 | lr=0.231, l2=1.88e-07, l1=8.11e-09 |
| 3 | update rules | Momentum | 8.3.2 | full labels | 20 | 0.9823 | 0.9806 ± 0.0014 | 0.99971 | 0.0935 | lr=0.05, l2=1e-05, l1=1e-07, alpha=0.9 |
| 4 | update rules | Nesterov momentum | 8.3.3 | full labels | 20 | 0.9787 | 0.9791 ± 0.0014 | 0.99968 | 0.0915 | lr=0.115, l2=1.88e-07, l1=8.11e-09, alpha=0.59 |
| 5 | update rules | Learning-rate decay (schedule within SGD) | 8.3.1 | full labels | 20 | 0.9755 | 0.9780 ± 0.0015 | 0.99961 | 0.0724 | lr=0.327, l2=0.000274, l1=6.67e-05, schedule=exponential, tau_frac=0.8, final_frac=0.01, gamma=0.702, d=0.5, step_epochs=5 |
| 6 | adaptive learning rate | AdaGrad | 8.5.1 | full labels | 20 | 0.9786 | 0.9822 ± 0.0013 | 0.99973 | 0.0591 | lr=0.017, l2=0.000514, l1=9.96e-09 |
| 7 | adaptive learning rate | RMSProp | 8.5.2 | full labels | 20 | 0.9788 | 0.9801 ± 0.0014 | 0.99967 | 0.0689 | lr=0.000817, l2=0.000514, l1=9.96e-09, rho=0.9 |
| 8 | adaptive learning rate | RMSProp with Nesterov momentum | 8.5.2 | full labels | 20 | 0.9786 | 0.9764 ± 0.0015 | 0.99947 | 0.1597 | lr=0.000448, l2=6.87e-08, l1=2.89e-08, rho=0.871, alpha=0.705 |
| 9 | adaptive learning rate | Adam | 8.5.3 | full labels | 20 | 0.9756 | 0.9764 ± 0.0015 | 0.99964 | 0.1102 | lr=0.000487, l2=1.49e-06, l1=9.91e-07, rho1=0.85, rho2=0.994 |
| 10 | second-order | Newton's method (damped, truncated / Hessian-free) | 8.6.1 | full labels | 437 | 0.9768 | 0.9773 ± 0.0015 | 0.99965 | 0.1128 | l2=8.63e-08, l1=6.03e-09, damping=0.00182, cg_iters=36 |
| 11 | second-order | Conjugate gradients (nonlinear, PR+/FR) | 8.6.2 | full labels | 214 | 0.9784 | 0.9791 ± 0.0014 | 0.99968 | 0.0674 | l2=1.77e-06, l1=5.67e-05, variant=PR+, restart=39, c2=0.12 |
| 12 | second-order | BFGS (dense inverse-Hessian, reduced network) | 8.6.3 | reduced network 784-8-10 (dense n x n matrix) | 136 | 0.8388 | 0.8477 ± 0.0036 | 0.98002 | 0.5249 | l2=3.91e-05, l1=6.03e-09, c2=0.57 |
| 13 | second-order | L-BFGS (limited-memory BFGS) | 8.6.3 | full labels | 129 | 0.9745 | 0.9737 ± 0.0016 | 0.99954 | 0.0874 | l2=0.00469, l1=2.64e-06, history=5, c2=0.508 |
| 14 | initialization | Random Gaussian / uniform initialization | 8.4 (12.1) | full labels | 20 | 0.9816 | 0.9822 ± 0.0013 | 0.99975 | 0.0735 | lr=0.0365, l2=5.59e-07, l1=1.15e-06, dist=uniform, std=0.00975 |
| 15 | initialization | Fixed-scale heuristic U(-1/sqrt(m), 1/sqrt(m)) | 8.4 (12.2) | full labels | 20 | 0.9826 | 0.9842 ± 0.0012 | 0.99978 | 0.0759 | lr=0.05, l2=1e-05, l1=1e-07, gain=1 |
| 16 | initialization | Normalized (Glorot/Xavier) initialization | 8.4 (12.3) | full labels | 20 | 0.9827 | 0.9819 ± 0.0013 | 0.99978 | 0.0765 | lr=0.05, l2=1e-05, l1=1e-07, gain=1 |
| 17 | initialization | Orthogonal initialization with gain (Saxe et al.) | 8.4 (12.4) | full labels | 20 | 0.9811 | 0.9801 ± 0.0014 | 0.99978 | 0.0798 | lr=0.05, l2=1e-05, l1=1e-07, gain=1.41 |
| 18 | initialization | Gain-tuned random-walk initialization (Sussillo) | 8.4 (12.5) | full labels | 20 | 0.9801 | 0.9813 ± 0.0014 | 0.99976 | 0.0666 | lr=0.0311, l2=0.000139, l1=2.54e-08, ratio=1.12 |
| 19 | initialization | Sparse initialization (k non-zero weights per unit) | 8.4 (12.6) | full labels | 20 | 0.9820 | 0.9829 ± 0.0013 | 0.99973 | 0.0918 | lr=0.0837, l2=6.87e-08, l1=2.89e-08, k=8, std=0.269 |
| 20 | initialization | Initial scale as a hyperparameter (search / LSUV) | 8.4 (12.7) | full labels | 20 | 0.9814 | 0.9840 ± 0.0013 | 0.99976 | 0.0598 | lr=0.0797, l2=0.000177, l1=1.27e-09, mode=search, scale_0=0.355, scale_1=0.327, scale_out=0.329 |
| 21 | initialization | Zero biases | 8.4 (12.8) | full labels | 20 | 0.9807 | 0.9823 ± 0.0013 | 0.99977 | 0.0558 | lr=0.0408, l2=0.000514, l1=9.96e-09 |
| 22 | initialization | Output bias = training-data marginal (softmax(b) = class frequencies) | 8.4 (12.9) | full labels | 20 | 0.9793 | 0.9809 ± 0.0014 | 0.99972 | 0.0685 | lr=0.0262, l2=0.000173, l1=3.85e-08, strength=0.018, out_weight_scale=0.269 |
| 23 | initialization | Small positive ReLU bias | 8.4 (12.10) | full labels | 20 | 0.9812 | 0.9834 ± 0.0013 | 0.99980 | 0.0529 | lr=0.0408, l2=0.000514, l1=9.96e-09, bias=0.257 |
| 24 | initialization | Gate biases near 1 (gated hidden units) | 8.4 (12.11) | gated hidden layers | 20 | 0.9806 | 0.9812 ± 0.0014 | 0.99977 | 0.0584 | lr=0.0408, l2=0.000514, l1=9.96e-09, gate_bias=1.09 |
| 25 | initialization | Variance/precision parameter init (Gaussian output, learned precision) | 8.4 (12.12) | Gaussian-output model (squared error + learned precision) | 20 | 0.9820 | 0.9809 ± 0.0014 | 0.99265 | 3.8828 | lr=0.000847, l2=0.00506, l1=4.57e-06, precision_init=one |
| 26 | initialization | Unsupervised (autoencoder) pretraining as initialization | 8.4 (12.13) | full labels | 20 | 0.9789 | 0.9806 ± 0.0014 | 0.99968 | 0.0735 | lr=0.0291, l2=0.000157, l1=4.2e-08, pre_frac=0.135, pre_lr=0.0762 |
| 27 | initialization | Supervised pretraining on a related / unrelated task | 8.4 (12.14) | full labels | 20 | 0.9798 | 0.9771 ± 0.0015 | 0.99957 | 0.1041 | lr=0.027, l2=3.61e-06, l1=1.6e-07, task=related, pre_frac=0.283, pre_lr=0.0654 |
| 28 | gradient clipping | Gradient clipping: element-wise (value) | 8.2.4 (13.1) | full labels | 20 | 0.9814 | 0.9823 ± 0.0013 | 0.99976 | 0.0567 | lr=0.0408, l2=0.000514, l1=9.96e-09, v=0.0349 |
| 29 | gradient clipping | Gradient clipping: norm | 8.2.4 (13.2) | full labels | 20 | 0.9820 | 0.9834 ± 0.0013 | 0.99975 | 0.0558 | lr=0.0408, l2=0.000514, l1=9.96e-09, v=0.762 |
| 30 | gradient clipping | Gradient clipping: back-propagated gradient w.r.t. hidden units | 8.2.4 (13.3) | full labels | 20 | 0.9815 | 0.9806 ± 0.0014 | 0.99973 | 0.0849 | lr=0.05, l2=1e-05, l1=1e-07, v=0.005 |
| 31 | meta-algorithms | Batch normalization | 8.7.1 | BatchNorm before each ReLU | 20 | 0.9843 | 0.9824 ± 0.0013 | 0.99981 | 0.0628 | lr=0.0731, l2=5.59e-07, l1=1.15e-06 |
| 32 | meta-algorithms | Block coordinate descent (one layer at a time) | 8.7.2 | full labels | 20 | 0.9753 | 0.9783 ± 0.0015 | 0.99963 | 0.0762 | lr=0.023, l2=0.00279, l1=9.32e-06, cycle=2 |
| 33 | meta-algorithms | Polyak averaging (exponential moving average of weights) | 8.7.3 | full labels | 20 | 0.9810 | 0.9829 ± 0.0013 | 0.99978 | 0.0552 | lr=0.0408, l2=0.000514, l1=9.96e-09, one_minus_tau=0.00498 |
| 34 | meta-algorithms | Supervised greedy layer-wise pretraining | 8.7.4 | full labels | 20 | 0.9802 | 0.9811 ± 0.0014 | 0.99969 | 0.0624 | lr=0.0263, l2=0.000158, l1=7.04e-05, pre_frac=0.629, pre_lr=0.0699, freeze=False |
| 35 | meta-algorithms | Model design: near-linear activations (ReLU, leaky ReLU, ELU, maxout vs tanh, sigmoid) | 8.7.5 (a) | full labels | 20 | 0.9808 | 0.9812 ± 0.0014 | 0.99973 | 0.0848 | lr=0.05, l2=1e-05, l1=1e-07, activation=relu |
| 36 | meta-algorithms | Model design: skip connections (deep residual MLP) | 8.7.5 (b) | deep MLP, 3 equal-width layers | 20 | 0.9784 | 0.9777 ± 0.0015 | 0.99965 | 0.0937 | lr=0.0197, l2=2.8e-06, l1=5.45e-07, depth=3 |
| 37 | meta-algorithms | Model design: auxiliary heads (deep supervision) | 8.7.5 (c) | deep MLP, 6 layers, heads discarded after training | 20 | 0.9829 | 0.9840 ± 0.0013 | 0.99968 | 0.1074 | lr=0.05, l2=1e-05, l1=1e-07, depth=6, aux_weight=0.3 |
| 38 | meta-algorithms | Continuation method (annealed input blur) | 8.7.6 (a) | full labels | 20 | 0.9836 | 0.9838 ± 0.0013 | 0.99979 | 0.0624 | lr=0.0455, l2=8.19e-05, l1=3.41e-08, sigma0=1.22, anneal_frac=0.58 |
| 39 | meta-algorithms | Curriculum learning (easy-to-hard sampling) | 8.7.6 (b) | full labels | 20 | 0.9807 | 0.9811 ± 0.0014 | 0.99973 | 0.0680 | lr=0.0308, l2=9.85e-05, l1=2.11e-06, start_frac=0.243, pace_frac=0.205 |
