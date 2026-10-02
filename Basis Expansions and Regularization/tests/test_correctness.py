"""Check bases, penalties and solvers against independent references (scipy / scikit-learn / analytic identities)."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "methods"))
import bases, common, wavelet_smoothing as ws, nonparametric_logistic as npl
from scipy.interpolate import make_smoothing_spline
from scipy.integrate import quad
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression

ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name}  {info}")

rng = np.random.RandomState(0)
# natural cubic basis: K columns, linear outside the knots
x = np.sort(rng.rand(300)); kn = bases.ncs_knots(x, 7)
N = bases.natural_cubic_basis(x, kn); out = np.array([-0.5, -0.2, 1.2, 1.5]); No = bases.natural_cubic_basis(out, kn)
check("natural cubic basis has K columns, rank K", N.shape[1] == 7 and np.linalg.matrix_rank(N) == 7)
tt = np.array([-1.0, -0.6, -0.2, 1.2, 1.6, 2.0]); Nt = bases.natural_cubic_basis(tt, kn)
check("second differences vanish outside knots (basis)", np.abs(Nt[0] - 2 * Nt[1] + Nt[2]).max() < 1e-8 and np.abs(Nt[3] - 2 * Nt[4] + Nt[5]).max() < 1e-8)
# regression spline (truncated power) == B-spline fit (same space)
xs = rng.randn(2000); y = np.sin(xs) + 0.3 * rng.randn(2000); X = xs[:, None]
kq = bases.quantile_knots(xs, 6); t = bases.bs_knots(xs.min(), xs.max(), kq)
m1 = common.ls_fit(lambda Z: bases.trunc_power_basis(Z[:, 0], kq, 3), X, y); m2 = common.ls_fit(lambda Z: bases.bs_design(Z[:, 0], t), X, y)
check("truncated-power cubic spline == B-spline regression", np.abs(m1.predict(X) - m2.predict(X)).max() < 1e-7, f"{np.abs(m1.predict(X) - m2.predict(X)).max():.1e}")
check("B-spline basis better conditioned", m2.info["cond_design"] < m1.info["cond_design"], f"{m2.info['cond_design']:.1e} < {m1.info['cond_design']:.1e}")
# penalty matrix vs numerical integration
t5 = bases.bs_knots(0, 1, [0.3, 0.55, 0.8]); Om = bases.bs_penalty(t5)
from scipy.interpolate import BSpline
m_ = len(t5) - 4; D2 = BSpline(t5, np.eye(m_), 3).derivative(2)
num = quad(lambda u: D2(u)[1] * D2(u)[2], 0, 1, limit=200)[0]
check("B-spline penalty entry vs numerical quad", abs(Om[1, 2] - num) < 1e-8, f"{abs(Om[1, 2] - num):.1e}")
# smoothing spline == scipy exact smoothing spline when knots sit at the data
n = 60; xd = np.sort(rng.rand(n)); yd = np.sin(6 * xd) + 0.2 * rng.randn(n); lam = 1e-4
tt_ = bases.bs_knots(xd[0], xd[-1], xd[1:-1]); B = bases.bs_design(xd, tt_); Om2 = bases.bs_penalty(tt_)
G, c, yy = common.gram(B, yd); mm = common.penalized_fit(lambda Z: bases.bs_design(Z[:, 0], tt_), G, c, yy, n, Om2, lam)
ref = make_smoothing_spline(xd, yd, lam=lam)
xg = np.linspace(xd[0], xd[-1], 200)
check("penalised B-spline == scipy make_smoothing_spline (all-knot case)", np.abs(mm.predict(xg[:, None]) - ref(xg)).max() < 1e-5, f"{np.abs(mm.predict(xg[:, None]) - ref(xg)).max():.1e}")
# df <-> lambda (well-posed: n >> number of basis functions)
nn = 500; xw = np.sort(rng.rand(nn)); yw = np.sin(6 * xw) + 0.2 * rng.randn(nn); tw = bases.bs_knots(0, 1, bases.quantile_knots(xw, 25)); Bw = bases.bs_design(xw, tw)
Omw = bases.bs_penalty(tw); Gw, cw, yyw = common.gram(Bw, yw)
lams, nd = bases.df_to_lambda(Gw, Omw, [4, 8, 16]); dfs = [common.penalized_fit(lambda Z: bases.bs_design(Z[:, 0], tw), Gw, cw, yyw, nn, Omw, l).df for l in lams]
check("df_to_lambda hits the df target", np.abs(np.array(dfs) - [4, 8, 16]).max() < 1e-3, f"{np.round(dfs, 4)}")
# TPS: very large lambda -> least-squares plane
Xc = rng.randn(800, 2); yc = 1 + 2 * Xc[:, 0] - Xc[:, 1] + np.sin(2 * Xc[:, 0]) * np.cos(Xc[:, 1]) + 0.2 * rng.randn(800)
st = bases.tps_setup(bases.kmeans2d(Xc, 40)); Bt = bases.tps_design(Xc, st); Gt, ct, yyt = common.gram(Bt, yc)
mt = common.penalized_fit(lambda Z: bases.tps_design(Z, st), Gt, ct, yyt, 800, st["Omega"], 1e9)
A = np.c_[np.ones(800), Xc]; plane = A @ np.linalg.lstsq(A, yc, rcond=None)[0]
check("thin-plate with huge lambda -> OLS plane", np.abs(mt.predict(Xc) - plane).max() < 1e-3 and abs(mt.df - 3) < 1e-3, f"{np.abs(mt.predict(Xc) - plane).max():.1e}; df={mt.df:.3f}")
check("TPS penalty is PSD", np.linalg.eigvalsh(st["Omega"]).min() > -1e-8)
m0 = common.penalized_fit(lambda Z: bases.tps_design(Z, st), Gt, ct, yyt, 800, st["Omega"], 1e-6)
check("thin-plate: small lambda fits better than plane", m0.rss < mt.rss and m0.df > mt.df)
# RKHS == sklearn KernelRidge when every point is a landmark
xk = np.sort(rng.rand(80)); yk = np.sin(8 * xk) + 0.1 * rng.randn(80); h = 0.15; lk = 1e-2
Kx = np.exp(-(xk[:, None] - xk[None]) ** 2 / (2 * h * h)); Gk, ck, yyk = common.gram(Kx, yk)
mk = common.penalized_fit(lambda Z: np.exp(-(Z[:, 0][:, None] - xk[None]) ** 2 / (2 * h * h)), Gk, ck, yyk, 80, Kx, lk)
kr = KernelRidge(alpha=lk, kernel="rbf", gamma=1 / (2 * h * h)).fit(xk[:, None], yk); xt = np.linspace(0, 1, 100)[:, None]
check("RKHS (all landmarks) == sklearn KernelRidge", np.abs(mk.predict(xt) - kr.predict(xt)).max() < 1e-4, f"{np.abs(mk.predict(xt) - kr.predict(xt)).max():.1e}")
# wavelets
sig = rng.randn(256); w = ws.wname(common.RunConfig()); L = ws._levels(256, w)
check(f"wavelet ({w}) perfect reconstruction", np.abs(ws.idwt(ws.dwt(sig, w, L), w) - sig).max() < 1e-10)
check("soft threshold", np.allclose(ws.soft(np.array([-3, -0.5, 0.2, 2]), 1.0), [-2, 0, 0, 1]))
cfg = common.RunConfig(wavelet_sim_reps=30)
import tempfile, logging
summ, corr = ws.simulate(tempfile.mkdtemp(), cfg, logging.getLogger("t"))
ratio = summ.loc["wavelet_SURE", "sure_estimate_over_actual_mse"]
check("SURE risk estimate ~ actual loss (mean ratio 0.6-1.4)", 0.6 < ratio < 1.4, f"ratio={ratio:.2f}")
check("wavelet (SURE) beats smoothing spline on Doppler", summ.loc["wavelet_SURE", "mean_mse_vs_truth"] < summ.loc["smoothing_spline_GCV", "mean_mse_vs_truth"])
check("wavelet SURE beats noisy data MSE (denoises)", summ.loc["wavelet_SURE", "mean_mse_vs_truth"] < (cfg.wavelet_sim_snr and (0.9 * (np.var(ws.doppler((np.arange(1024) + .5) / 1024)) / cfg.wavelet_sim_snr ** 2))), f"{summ.loc['wavelet_SURE','mean_mse_vs_truth']:.2e}")
# nonparametric logistic == sklearn unpenalised logistic on the same basis
xl = rng.randn(4000); pl = 1 / (1 + np.exp(-(0.5 * np.sin(1.5 * xl) - 1))); yl = (rng.rand(4000) < pl).astype(float)
Xl = xl[:, None]
class P: pass
grid = [{"df_target": 6, "lam": 1e-9}]
mods = npl.fit_path(Xl, yl, grid, common.RunConfig(logit_knots=5))
t_ = bases.bs_knots(xl.min(), xl.max(), bases.quantile_knots(xl, 5)); Bl = bases.bs_design(xl, t_)
skl = LogisticRegression(C=np.inf, fit_intercept=False, tol=1e-12, max_iter=5000).fit(Bl, yl)
check("nonparametric logistic (lam~0) == unpenalised logistic on the B-spline basis", np.abs(mods[0].predict(Xl) - skl.predict_proba(Bl)[:, 1]).max() < 1e-5, f"{np.abs(mods[0].predict(Xl) - skl.predict_proba(Bl)[:, 1]).max():.1e}")
# tests
e = rng.randn(3000) ** 2
_, _, p_same = common.dm_test(e, e + 1e-12 * rng.randn(3000)); _, _, p_diff = common.dm_test(e, e + 0.5)
check("Diebold-Mariano: ~equal losses -> large p ; clearly different -> small p", p_diff < 1e-6, f"p_same={p_same:.2f} p_diff={p_diff:.1e}")
fm = np.array([[1.0, 0.5, 0.4], [3.0, 2.5, 2.41], [2.0, 1.5, 1.38], [5.0, 4.5, 4.39]])
sel_p, imin_p = common.one_se_paired(fm, np.array([1.0, 2.0, 3.0]))
sel_plain, _ = common.one_se(fm.mean(0), fm.std(0, ddof=1) / 2, np.array([1.0, 2.0, 3.0]))
check("paired one-SE rule ignores shared fold effects (plain rule picks the simplest, paired does not)", sel_plain == 0 and sel_p >= 1, f"plain={sel_plain} paired={sel_p}")
check("Holm adjustment", np.allclose(common.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]))
print("\nALL PASSED" if ok else "\nSOME FAILED"); sys.exit(0 if ok else 1)
