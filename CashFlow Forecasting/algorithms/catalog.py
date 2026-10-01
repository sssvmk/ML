"""
Catalog of all 38 algorithm modules (PRD §3.2 is the definitive
manifest of which 38). Maps each algorithm's id to where its class
lives, so config.json can enable/configure algorithms by id without the
orchestrator hard-coding imports (Design doc §5.1: the Orchestrator only
knows the nine interfaces, never a module's internals -- this catalog is
the one place that also knows import paths, for wiring purposes only).
"""

# id -> (module path under `algorithms`, class name, PRD §3.2 row #)
# PRD row #30 (PatchTSMixer) is DROPPED entirely (your decision): the IBM-checkpoint adapter is removed, not just left
# unconfigured. The catalog therefore has 37 entries, one short of the PRD's "38" -- part of gap G-40 (PRD wording needs the
# PRD owner); row numbers below are kept as historical PRD references, not renumbered.
ALGORITHM_CATALOG: dict[str, tuple[str, str, int]] = {
    "arimax": ("algorithms.arimax", "ARIMAXModule", 1),
    "sarimax": ("algorithms.sarimax", "SARIMAXModule", 2),
    "varmax": ("algorithms.varmax", "VARMAXModule", 3),
    "dynamic_regression": ("algorithms.dynamic_regression", "DynamicRegressionModule", 4),
    "state_space": ("algorithms.state_space", "StateSpaceModule", 5),
    "structural_ts": ("algorithms.structural_ts", "StructuralTSModule", 6),
    "prophet": ("algorithms.prophet_model", "ProphetModule", 7),
    "linear_regression": ("algorithms.linear_regression", "LinearRegressionModule", 8),
    "polynomial_regression": ("algorithms.polynomial_regression", "PolynomialRegressionModule", 9),
    "ridge": ("algorithms.ridge", "RidgeModule", 10),
    "lasso": ("algorithms.lasso", "LassoModule", 11),
    "elastic_net": ("algorithms.elastic_net", "ElasticNetModule", 12),
    "random_forest": ("algorithms.random_forest", "RandomForestModule", 13),
    "extra_trees": ("algorithms.extra_trees", "ExtraTreesModule", 14),
    "xgboost": ("algorithms.xgboost_module", "XGBoostModule", 15),
    "lightgbm": ("algorithms.lightgbm_module", "LightGBMModule", 16),
    "catboost": ("algorithms.catboost_module", "CatBoostModule", 17),
    "svr": ("algorithms.svr", "SVRModule", 18),
    "knn_regression": ("algorithms.knn_regression", "KNNRegressionModule", 19),
    "rnn": ("algorithms.rnn", "RNNModule", 20),
    "lstm": ("algorithms.lstm", "LSTMModule", 21),
    "gru": ("algorithms.gru", "GRUModule", 22),
    "tcn": ("algorithms.tcn", "TCNModule", 23),
    "wavenet": ("algorithms.wavenet", "WaveNetModule", 24),
    "deepar": ("algorithms.deepar", "DeepARModule", 25),
    "deepstate": ("algorithms.deepstate", "DeepStateModule", 26),
    "deepvar": ("algorithms.deepvar", "DeepVARModule", 27),
    "tft": ("algorithms.tft", "TFTModule", 28),
    "tide": ("algorithms.tide", "TiDEModule", 29),
    "timesnet": ("algorithms.timesnet", "TimesNetModule", 31),
    "itransformer": ("algorithms.itransformer", "ITransformerModule", 32),
    "informer": ("algorithms.informer", "InformerModule", 33),
    "autoformer": ("algorithms.autoformer", "AutoformerModule", 34),
    "fedformer": ("algorithms.fedformer", "FEDformerModule", 35),
    "etsformer": ("algorithms.etsformer", "ETSformerModule", 36),
    "nbeats": ("algorithms.nbeats", "NBEATSModule", 37),
    "nhits": ("algorithms.nhits", "NHiTSModule", 38),
}

assert len(ALGORITHM_CATALOG) == 37, f"expected 37 algorithm modules (38 minus dropped PatchTSMixer, G-40/G-30), catalog has {len(ALGORITHM_CATALOG)}"


# Optional roster additions OUTSIDE the PRD §3.2 manifest of 38 (matrix roster item #39). Kept separate so the PRD count above stays
# exactly what the PRD says; whether the catalog count becomes configuration-dependent is gap G-40 and needs the PRD owner.
EXTENSION_CATALOG: dict[str, tuple[str, str, int]] = {
    "timesfm3": ("algorithms.timesfm3", "TimesFM3Module", 39),
}


def load_class(algorithm_id: str):
    import importlib
    module_path, class_name, _ = {**ALGORITHM_CATALOG, **EXTENSION_CATALOG}[algorithm_id]
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)
