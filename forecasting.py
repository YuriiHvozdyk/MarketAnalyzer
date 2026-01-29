import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class PriceForecaster:
    """
    Forecast future prices using multiple models
    and select the best one automatically.
    """

    def __init__(self, df: pd.DataFrame, target: str = "Close"):
        if target not in df.columns:
            raise ValueError("Target column not found.")

        self.df = df.copy()
        self.target = target

    def _prepare_data(self):
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        return train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

    def train_and_select_model(self):
        X_train, X_test, y_train, y_test = self._prepare_data()

        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(
                n_estimators=200,
                random_state=42
            )
        }

        best_model = None
        best_rmse = float("inf")

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                self.best_model_name = name

        self.model = best_model
        self.rmse = best_rmse
        return best_model, best_rmse

    def predict_next(self, X_last: pd.DataFrame) -> float:
        if not hasattr(self, "model"):
            raise RuntimeError("Model is not trained.")
        return float(self.model.predict(X_last)[0])
