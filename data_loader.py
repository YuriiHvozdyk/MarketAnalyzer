"""
Module for loading historical stock market data.

This module provides a class for downloading historical stock price data
from Yahoo Finance using the yfinance library.
"""

from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf


class MarketDataLoader:
    """
    A class for loading historical market data for a given stock ticker.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. 'AAPL', 'MSFT').

    Examples
    --------
    >>> loader = MarketDataLoader("AAPL")
    >>> data = loader.load_by_period("1y")
    >>> data.head()
    """

    _ALLOWED_PERIODS = {
        "1d", "5d", "1mo", "3mo", "6mo",
        "1y", "2y", "5y", "10y", "ytd", "max"
    }

    def __init__(self, ticker: str) -> None:
        self._validate_ticker(ticker)
        self.ticker = ticker.upper()

    @staticmethod
    def _validate_ticker(ticker: str) -> None:
        if not isinstance(ticker, str):
            raise TypeError("Ticker must be a string.")
        if not ticker.strip():
            raise ValueError("Ticker cannot be empty.")

    @staticmethod
    def _validate_dates(start_date: str, end_date: str) -> None:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(
                "Dates must be in 'YYYY-MM-DD' format."
            ) from exc

        if start >= end:
            raise ValueError("Start date must be earlier than end date.")

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("No data returned for the given parameters.")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required_columns = {"Open", "High", "Low", "Close", "Volume"}
        if not required_columns.issubset(set(df.columns)):
            raise ValueError(
                f"Downloaded data does not contain required columns: {required_columns}"
            )

    def load_by_period(self, period: str) -> pd.DataFrame:
        if period not in self._ALLOWED_PERIODS:
            raise ValueError(
                f"Invalid period '{period}'. Allowed values: {self._ALLOWED_PERIODS}"
            )

        df = yf.download(
            tickers=self.ticker,
            period=period,
            auto_adjust=False,
            progress=False
        )

        self._validate_dataframe(df)
        return df

    def load_by_dates(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        self._validate_dates(start_date, end_date)

        df = yf.download(
            tickers=self.ticker,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False
        )

        self._validate_dataframe(df)
        return df
