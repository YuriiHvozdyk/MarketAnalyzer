import pandas as pd
import numpy as np


class FeatureEngineer:
    """
    Feature engineering for financial time series.
    Allows chainable transformations for convenience.
    """

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        self.df = df.copy()

    def add_lag_features(self, lags: list[int]) -> "FeatureEngineer":
        """
        Add lag features of the 'Close' price.

        Parameters
        ----------
        lags : list[int]
            List of lag periods (positive integers).

        Returns
        -------
        self : FeatureEngineer
            Returns self to allow chaining.
        """
        for lag in lags:
            if lag <= 0:
                raise ValueError("Lag values must be positive.")
            self.df[f"Close_lag_{lag}"] = self.df["Close"].shift(lag)
        return self

    def add_returns(self) -> "FeatureEngineer":
        """
        Add simple returns column: Return_t = (Close_t / Close_t-1) - 1

        Returns
        -------
        self : FeatureEngineer
        """
        self.df["Return"] = self.df["Close"].pct_change()
        return self

    def add_volatility(self, window: int = 10) -> "FeatureEngineer":
        """
        Add rolling volatility of returns.

        Parameters
        ----------
        window : int
            Rolling window for standard deviation calculation.

        Returns
        -------
        self : FeatureEngineer
        """
        if window <= 0:
            raise ValueError("Window must be positive.")
        if "Return" not in self.df.columns:
            self.add_returns()
        self.df["Volatility"] = self.df["Return"].rolling(window).std()
        return self

    def drop_na(self) -> pd.DataFrame:
        """
        Drop rows with missing values.

        Returns
        -------
        pandas.DataFrame
            Cleaned DataFrame with no NaNs.
        """
        self.df.dropna(inplace=True)
        return self.df
