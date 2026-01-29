import pandas as pd
import numpy as np


class TechnicalIndicators:
    """
    A collection of technical analysis indicators calculated
    based on OHLCV market data.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the TechnicalIndicators class.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing market data with at least
            a 'Close' price column.
        """
        self.data = data.copy()
        self._validate()

    def _validate(self) -> None:
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")

        if "Close" not in self.data.columns:
            raise ValueError("DataFrame must contain a 'Close' column.")

        if self.data.empty:
            raise ValueError("Input DataFrame is empty.")

    def sma(self, window: int) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA).

        SMA = (P1 + P2 + ... + Pn) / n

        Parameters
        ----------
        window : int
            Number of periods for averaging.

        Returns
        -------
        pandas.Series
            Simple moving average values.

        Examples
        --------
        >>> indicators = TechnicalIndicators(df)
        >>> indicators.sma(20)
        """
        if not isinstance(window, int) or window <= 0:
            raise ValueError("Window must be a positive integer.")

        return self.data["Close"].rolling(window=window).mean()

    def ema(self, window: int) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).

        Parameters
        ----------
        window : int
            Number of periods for EMA.

        Returns
        -------
        pandas.Series
            Exponential moving average values.
        """
        if not isinstance(window, int) or window <= 0:
            raise ValueError("Window must be a positive integer.")

        return self.data["Close"].ewm(span=window, adjust=False).mean()

    def rsi(self, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        RSI = 100 - (100 / (1 + RS))

        Parameters
        ----------
        window : int, optional
            Lookback period, by default 14.

        Returns
        -------
        pandas.Series
            RSI values.
        """
        if not isinstance(window, int) or window <= 0:
            raise ValueError("Window must be a positive integer.")

        delta = self.data["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def macd(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """
        Calculate Moving Average Convergence Divergence (MACD).

        Parameters
        ----------
        fast : int
            Fast EMA period.
        slow : int
            Slow EMA period.
        signal : int
            Signal line EMA period.

        Returns
        -------
        pandas.DataFrame
            DataFrame with MACD line, signal line and histogram.
        """
        if not all(isinstance(x, int) and x > 0 for x in [fast, slow, signal]):
            raise ValueError("All periods must be positive integers.")

        ema_fast = self.ema(fast)
        ema_slow = self.ema(slow)

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame({
            "MACD": macd_line,
            "Signal": signal_line,
            "Histogram": histogram
        })
