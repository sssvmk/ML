"""
MLflow pyfunc wrapper: the registered MLP model is the whole inference unit
(preprocessing + network + post-processing), not just raw weights.

  housing : input = DataFrame with the 8 raw feature columns
            output = DataFrame[prediction_100k, prediction_usd]
  csv     : (models trained with run.py --csv-path) input = DataFrame with the ORIGINAL feature
            columns; the stored preprocessing is applied inside the model.
            output = classification: DataFrame[label, confidence, prob_<class>..., out_of_range_warning]
                     regression:     DataFrame[prediction, out_of_range_warning]
  mnist   : input = float32 array [n,784] (or DataFrame of 784 columns), pixels scaled 0..1,
            white digit on black
            output = DataFrame[label, confidence, prob_0 ... prob_9]

Serve it with:  mlflow models serve -m models:/MLP-housing@champion
"""
import mlflow.pyfunc
import numpy as np
import pandas as pd


class MLPModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from infer import Predictor  # shipped with the model through code_paths
        self.predictor = Predictor(context.artifacts["bundle"])

    def predict(self, context, model_input, params=None):
        p = self.predictor
        if p.pre.get("type") == "tabular":
            df = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
            return p.tabular_frame(df)
        if p.task == "regression":
            names = p.pre["feature_names"]
            if isinstance(model_input, pd.DataFrame):
                missing = [n for n in names if n not in model_input.columns]
                if missing:
                    raise ValueError(f"missing input columns: {missing}")
                X = model_input[names].to_numpy(dtype=np.float32)
            else:
                X = np.asarray(model_input, dtype=np.float32)
            res = p.predict_housing(X)
            return pd.DataFrame({
                "prediction_100k": [r["prediction_100k"] for r in res],
                "prediction_usd": [r["prediction_usd"] for r in res],
            })

        X = model_input.to_numpy(dtype=np.float32) if isinstance(model_input, pd.DataFrame) \
            else np.asarray(model_input, dtype=np.float32)
        probs = p.mnist_probs(X)
        out = pd.DataFrame({
            "label": probs.argmax(axis=1).astype(np.int64),
            "confidence": probs.max(axis=1).astype(np.float64),
        })
        for i in range(probs.shape[1]):
            out[f"prob_{i}"] = probs[:, i].astype(np.float64)
        return out
