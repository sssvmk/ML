# STC patch record for the vendored NeuralForecast

- Upstream: Nixtla/neuralforecast, version **3.2.2**, licence **Apache-2.0** (see `LICENSES_neuralforecast/LICENSE`).
- Reason (decision D-9, strategy S2): every released NeuralForecast imports Ray at module level, so
  `import neuralforecast` fails where Ray is not installed (Windows-native runtimes).
- Changes (Apache-2.0 §4(b): modified files carry a prominent notice; these are the only modified files):

| File | Change |
|---|---|
| `neuralforecast/__init__.py` | `__version__` falls back to `"3.2.2+stc-lazy-ray"` when the package is not pip-installed |
| `neuralforecast/auto.py` | ray imports wrapped in try/except (names become `None`) |
| `neuralforecast/common/_base_auto.py` | ray imports wrapped in try/except; `_require_ray()` raises a clear error only when an `Auto*`/ray backend is actually used; default `search_alg` resolved lazily |

- Not changed: every model file (`models/*.py`), `core.py`, `tsdataset.py`, losses. This system never uses the
  `Auto*` classes (search.py performs the hyperparameter search), so the Ray-dependent paths are unreachable here.
- Optuna is also not required for plain models.
- Verification: `python -c "import neuralforecast"` succeeds with ray and optuna both uninstalled; tests/test_neural_nf.py.
