import pandas as pd
import matplotlib.pyplot as plt


def _validate_dataframe(df: pd.DataFrame, required_columns: set) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"DataFrame must contain columns: {required_columns}"
        )


def plot_price_with_ma(
    df: pd.DataFrame,
    ma_columns: list[str],
    title: str = "Price with Moving Averages"
) -> None:
    """
    Plot closing price along with moving averages.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'Close' price and MA columns.
    ma_columns : list of str
        Column names of moving averages.
    title : str
        Plot title.
    """
    _validate_dataframe(df, {"Close", *ma_columns})

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Close"], label="Close Price")

    for col in ma_columns:
        plt.plot(df.index, df[col], label=col)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_rsi(
    df: pd.DataFrame,
    rsi_column: str = "RSI",
    title: str = "Relative Strength Index (RSI)"
) -> None:
    """
    Plot RSI indicator with overbought/oversold levels.
    """
    _validate_dataframe(df, {rsi_column})

    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df[rsi_column], label="RSI")

    plt.axhline(70, linestyle="--")
    plt.axhline(30, linestyle="--")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_macd(
    macd_df: pd.DataFrame,
    title: str = "MACD Indicator"
) -> None:
    """
    Plot MACD line, signal line and histogram.
    """
    _validate_dataframe(macd_df, {"MACD", "Signal", "Histogram"})

    plt.figure(figsize=(12, 5))

    plt.plot(macd_df.index, macd_df["MACD"], label="MACD")
    plt.plot(macd_df.index, macd_df["Signal"], label="Signal")

    plt.bar(
        macd_df.index,
        macd_df["Histogram"],
        alpha=0.3,
        label="Histogram"
    )

    plt.title(title)
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True)
    plt.show()
