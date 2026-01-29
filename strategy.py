class TradingStrategy:
    """
    Simple decision based on predicted price vs last close
    and optional confidence threshold.
    """

    def __init__(self, last_close: float, predicted_price: float, threshold: float = 0.01):
        """
        Parameters
        ----------
        last_close : float
            Last closing price.
        predicted_price : float
            Price forecast from model.
        threshold : float
            Minimum relative change to trigger BUY/SELL.
        """
        self.last_close = last_close
        self.predicted_price = predicted_price
        self.threshold = threshold

    def recommend(self) -> str:
        """
        Returns:
            'BUY', 'SELL' or 'HOLD'
        """
        change = (self.predicted_price - self.last_close) / self.last_close

        if change > self.threshold:
            return "BUY"
        elif change < -self.threshold:
            return "SELL"
        else:
            return "HOLD"
