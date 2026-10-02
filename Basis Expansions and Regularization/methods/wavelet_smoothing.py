"""
Wavelet smoothing (ESLII 5.9).  Objective: min ||y - W theta||^2 + 2*lambda*||theta||_1 over an orthonormal wavelet basis W  ->  soft-thresholding
of the detail coefficients, threshold near sigma*sqrt(2 log N) (VisuShrink) or chosen by SURE (SureShrink).
Zillow application (as agreed): the chosen predictor is cut into N = 2^k equal-width bins; the bin means of logerror form a regular 1-D signal
(empty bins interpolated); the signal is wavelet-transformed, soft-thresholded, inverted, and predictions are bin look-ups.
Selection: threshold chosen by 10-fold CV on TRAIN (common protocol); the SURE-chosen threshold is reported next to it.
Metrics: CV MSE vs threshold (one-SE rule on the number of non-zero coefficients), SURE risk curve, # non-zero coefficients, SNR, test MSE +- SE.
Also runs a SIMULATED Doppler experiment (simulation/): MSE against the TRUE signal, SURE risk estimate vs actual loss, sparsity.
"""
import _bootstrap  # noqa: F401
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import common
from common import Smoother

try:
    import pywt
    HAVE_PYWT = True
except Exception:                                      # pragma: no cover
    HAVE_PYWT = False

NAME = "wavelet_smoothing"


def wname(cfg):
    return cfg.wavelet if HAVE_PYWT else "haar"


def _levels(N, w):
    if HAVE_PYWT:
        return max(1, min(pywt.dwt_max_level(N, pywt.Wavelet(w).dec_len), int(np.log2(N)) - 3))
    return max(1, int(np.log2(N)) - 3)


def dwt(sig, w, level):
    if HAVE_PYWT:
        return pywt.wavedec(sig, w, mode="periodization", level=level)
    a, det = np.asarray(sig, float).copy(), []
    for _ in range(level):
        ev, od = a[0::2], a[1::2]
        det.append((ev - od) / np.sqrt(2)); a = (ev + od) / np.sqrt(2)
    return [a] + det[::-1]


def idwt(coeffs, w):
    if HAVE_PYWT:
        return pywt.waverec(coeffs, w, mode="periodization")
    a = coeffs[0]
    for d in coeffs[1:]:
        o = np.empty(2 * len(a)); o[0::2] = (a + d) / np.sqrt(2); o[1::2] = (a - d) / np.sqrt(2); a = o
    return a


def soft(c, t):
    return np.sign(c) * np.maximum(np.abs(c) - t, 0.0)


def sigma_hat(coeffs):
    return float(np.median(np.abs(coeffs[-1])) / 0.6745)


def sure_curve(details, sigma, ts):
    """SURE risk estimate (soft threshold) of the detail coefficients: sigma^2 [ N - 2 #{|s|<=t} + sum min(s,t)^2 ], s = c / sigma."""
    s = np.abs(np.concatenate(details)) / sigma
    return np.array([sigma ** 2 * (len(s) - 2 * np.sum(s <= t) + np.sum(np.minimum(s, t) ** 2)) for t in ts])


def _bin_means(xs, y, N, lo, hi):
    idx = np.clip(((xs - lo) / (hi - lo) * N).astype(int), 0, N - 1)
    cnt = np.bincount(idx, minlength=N).astype(float)
    sm = np.bincount(idx, weights=y, minlength=N)
    mean = np.full(N, np.nan)
    ok = cnt > 0
    mean[ok] = sm[ok] / cnt[ok]
    if not ok.all():
        mean = np.interp(np.arange(N), np.where(ok)[0], mean[ok])
    return mean, idx


def make_grid(prep, cfg, X, y):
    xs = X[:, 0]
    lo, hi = xs.min(), xs.max()
    for k in range(10, 4, -1):
        N = 2 ** k
        cnt = np.bincount(np.clip(((xs - lo) / (hi - lo) * N).astype(int), 0, N - 1), minlength=N)
        if np.mean(cnt >= 20) >= 0.8:
            break
    w = wname(cfg)
    mean, _ = _bin_means(xs, y, N, lo, hi)
    co = dwt(mean, w, _levels(N, w))
    sg = sigma_hat(co)
    ts = list(sg * np.geomspace(4.0, 0.02, 25)) + [0.0]
    return [{"N": N, "t": float(t), "t_over_sigma": float(t / sg), "sigma_hat": sg} for t in ts]


def fit_path(X, y, grid, cfg):
    xs = X[:, 0]
    lo, hi, N = xs.min(), xs.max(), grid[0]["N"]
    w = wname(cfg)
    mean, _ = _bin_means(xs, y, N, lo, hi)
    co = dwt(mean, w, _levels(N, w))
    out = []
    for hp in grid:
        sh = [co[0]] + [soft(c, hp["t"]) for c in co[1:]]
        curve = idwt(sh, w)
        nnz = int(sum((c != 0).sum() for c in sh[1:]) + len(sh[0]))
        pred = lambda Z, curve=curve: curve[np.clip(((Z[:, 0] - lo) / (hi - lo) * N).astype(int), 0, N - 1)]
        out.append(Smoother(pred, df=float(nnz), info={"threshold": hp["t"], "nonzero_coefficients": nnz, "bins": N, "wavelet": w}))
    return out


def extra(ctx):
    Xtr, y, grid, models, prep = ctx["Xtr"], ctx["y_fit_tr"], ctx["grid"], ctx["models"], ctx["prep"]
    cfg = ctx["cfg"]
    w, N = wname(cfg), grid[0]["N"]
    xs = Xtr[:, 0]
    mean, _ = _bin_means(xs, y, N, xs.min(), xs.max())
    co = dwt(mean, w, _levels(N, w))
    sg = sigma_hat(co)
    ts = np.array([g["t"] for g in grid])
    sure = sure_curve(co[1:], sg, ts)
    i_sure = int(np.argmin(sure))
    m = models[i_sure]
    val = float(np.mean((prep.y_val - m.predict(ctx["Xva"])) ** 2)); te = float(np.mean((prep.y_test - m.predict(ctx["Xte"])) ** 2))
    sel = ctx["summary"]["selected"]
    sig_var = float(np.var(m.predict(Xtr)))
    res = {"wavelet": w, "n_bins": N, "sigma_hat_bins": sg, "sure_threshold": float(ts[i_sure]), "sure_threshold_over_sigma": float(ts[i_sure] / sg),
           "universal_threshold": float(sg * np.sqrt(2 * np.log(N))), "cv_selected_threshold": sel["t"], "n_nonzero_at_sure": m.df, "n_nonzero_at_cv_choice": ctx["summary"]["selected_df"],
           "validation_mse_sure_choice_train_only": val, "test_mse_sure_choice_train_only_fit": te,
           "snr_db_signal_var_over_bin_noise_var": float(10 * np.log10(max(sig_var, 1e-300) / sg ** 2))}
    (ctx["out"] / "sure_vs_cv.json").write_text(json.dumps(common._jsonable(res), indent=2))
    ctx["summary"]["wavelet_extra"] = res
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ts[:-1] / sg, sure[:-1] / len(np.concatenate(co[1:])), "o-", ms=3, label="SURE risk estimate (detail coefficients, per coef.)")
    ax2 = ax.twinx(); ax2.plot(ts[:-1] / sg, ctx["curve"]["cv_mse"].values[:-1], "g.--", label="10-fold CV MSE")
    ax.set_xscale("log"); ax.set_xlabel("threshold / sigma_hat"); ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8); ax.set_title("Wavelet threshold: SURE vs CV")
    fig.tight_layout(); fig.savefig(ctx["out"] / "sure_curve.png", dpi=120); plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------
def doppler(x, eps=0.05):
    return np.sqrt(x * (1 - x)) * np.sin(2 * np.pi * (1 + eps) / (x + eps))


def simulate(out, cfg, log):
    from bases import bs_design, bs_knots, bs_penalty, df_to_lambda, quantile_knots
    from common import gram, penalized_fit
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    N, reps = cfg.wavelet_sim_n, cfg.wavelet_sim_reps
    x = (np.arange(N) + 0.5) / N
    f = doppler(x)
    sigma = f.std() / cfg.wavelet_sim_snr
    w = wname(cfg)
    L = _levels(N, w)
    t = bs_knots(0, 1, quantile_knots(x, 120))
    B, Om = bs_design(x, t), bs_penalty(t)
    tg = [d for d in (6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100) if d < B.shape[1] - 1]
    lams, _ = df_to_lambda(B.T @ B, Om, tg)
    rng = np.random.RandomState(cfg.seed)
    rows, keep = [], None
    for r in range(reps):
        y = f + sigma * rng.randn(N)
        co = dwt(y, w, L)
        sg = sigma_hat(co)
        det = np.concatenate(co[1:])
        cand = np.sort(np.abs(det) / sg)
        s = np.abs(det) / sg
        sure_all = np.array([len(s) - 2 * np.sum(s <= tt) + np.sum(np.minimum(s, tt) ** 2) for tt in cand[::max(1, len(cand) // 400)]])
        tt_grid = cand[::max(1, len(cand) // 400)]
        j = int(np.argmin(sure_all))
        t_sure, t_univ = sg * tt_grid[j], sg * np.sqrt(2 * np.log(N))
        fits = {}
        for nm, th in (("wavelet_SURE", t_sure), ("wavelet_universal", t_univ)):
            sh = [co[0]] + [soft(c, th) for c in co[1:]]
            fits[nm] = (idwt(sh, w), int(sum((c != 0).sum() for c in sh[1:])))
        sure_est = (sg ** 2) * (sure_all[j] + len(co[0])) / N
        # smoothing spline, df chosen by GCV
        G, c, yy = gram(B, y)
        best = None
        for lam in lams:
            m = penalized_fit(lambda Z: bs_design(Z[:, 0], t), G, c, yy, N, Om, lam)
            gcv = (m.rss / N) / (1 - m.df / N) ** 2
            if best is None or gcv < best[0]:
                best = (gcv, m)
        fits["smoothing_spline_GCV"] = (best[1].predict(x[:, None]), best[1].df)
        for nm, (fh, nz) in fits.items():
            rows.append({"rep": r, "method": nm, "mse_vs_truth": float(np.mean((fh - f) ** 2)), "nonzero_or_df": nz,
                         "sure_risk_estimate": sure_est if nm == "wavelet_SURE" else np.nan})
        if r == 0:
            keep = (y, {k: v[0] for k, v in fits.items()})
    df = pd.DataFrame(rows)
    df.to_csv(out / "simulation_results.csv", index=False)
    g = df.groupby("method")
    summ = pd.DataFrame({"mean_mse_vs_truth": g["mse_vs_truth"].mean(), "se": g["mse_vs_truth"].std() / np.sqrt(reps), "mean_nonzero_or_df": g["nonzero_or_df"].mean()})
    summ["signal_to_noise_snr"] = cfg.wavelet_sim_snr
    sw = df[df["method"] == "wavelet_SURE"]
    summ.loc["wavelet_SURE", "mean_sure_risk_estimate"] = sw["sure_risk_estimate"].mean()
    summ.loc["wavelet_SURE", "sure_estimate_over_actual_mse"] = sw["sure_risk_estimate"].mean() / sw["mse_vs_truth"].mean()
    summ.to_csv(out / "simulation_summary.csv")
    fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    y0, fit0 = keep
    for a, (nm, fh) in zip(ax, fit0.items()):
        a.plot(x, y0, ".", color="lightgrey", ms=2); a.plot(x, f, "k-", lw=1, label="truth"); a.plot(x, fh, "r-", lw=1.2, label=nm); a.legend(fontsize=8)
    fig.suptitle(f"Doppler, N={N}, SNR={cfg.wavelet_sim_snr}, wavelet={w}"); fig.tight_layout(); fig.savefig(out / "simulation_fits.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    df.boxplot(column="mse_vs_truth", by="method", ax=ax[0]); ax[0].set_title("MSE against the true signal"); ax[0].set_ylabel("MSE")
    ax[1].scatter(sw["sure_risk_estimate"], sw["mse_vs_truth"], s=12); mx = max(sw["sure_risk_estimate"].max(), sw["mse_vs_truth"].max())
    ax[1].plot([0, mx], [0, mx], "k:"); ax[1].set_xlabel("SURE risk estimate"); ax[1].set_ylabel("actual MSE vs truth"); ax[1].set_title("SURE estimates the true risk")
    fig.suptitle(""); fig.tight_layout(); fig.savefig(out / "simulation_mse_and_sure.png", dpi=120); plt.close(fig)
    corr = float(np.corrcoef(sw["sure_risk_estimate"], sw["mse_vs_truth"])[0, 1])
    log.info("simulation (%d reps): %s | mean SURE estimate / mean actual MSE = %.3f ; corr = %.3f", reps, summ["mean_mse_vs_truth"].round(6).to_dict(),
             summ.loc["wavelet_SURE", "sure_estimate_over_actual_mse"], corr)
    return summ, corr


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    s = common.run_smoother(NAME, prep_dir, out_dir, cfg, progress, inputs="x", make_grid=make_grid, fit_path=fit_path, extra=extra,
                            notes=f"wavelet={wname(cfg)} ({'PyWavelets' if HAVE_PYWT else 'built-in Haar fallback'}); bins = 2^k equal-width in the transformed predictor.", console=console)
    lg = common.get_logger(Path(out_dir) / "simulation", "simulation", console=console)
    summ, corr = simulate(Path(out_dir) / "simulation", cfg, lg)
    s["simulation"] = {"mean_mse_vs_truth": summ["mean_mse_vs_truth"].to_dict(), "sure_vs_actual_corr": corr,
                       "sure_estimate_over_actual_mse": float(summ.loc["wavelet_SURE", "sure_estimate_over_actual_mse"])}
    (Path(out_dir) / "summary.json").write_text(json.dumps(common._jsonable(s), indent=2))
    return s


if __name__ == "__main__":
    common.method_main(NAME, run)
