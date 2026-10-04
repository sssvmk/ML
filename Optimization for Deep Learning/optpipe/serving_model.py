"""MLflow pyfunc wrapper so the registered model can be served with `mlflow models serve`
or loaded with mlflow.pyfunc.load_model("models:/MNIST-Reg@champion")."""
import mlflow.pyfunc
import numpy as np
import pandas as pd


class MNISTOptModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from optpipe.bundle import Predictor
        self.predictor = Predictor(context.artifacts["bundle"])

    def predict(self, context, model_input, params=None):
        x = model_input.to_numpy() if hasattr(model_input, "to_numpy") else np.asarray(model_input)
        p = self.predictor.predict_proba(x.astype(np.float32))
        df = pd.DataFrame(p, columns=[f"prob_{i}" for i in range(p.shape[1])])
        df.insert(0, "confidence", p.max(1))
        df.insert(0, "label", p.argmax(1))
        return df
