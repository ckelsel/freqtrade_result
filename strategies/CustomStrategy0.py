# 优化下买点，实现populate_buy_trend，买点算法，
# 1）时间周期1小时
# 2）

# k线横盘超过10小时，振幅不超过8%
# k线横盘超过20小时，不断跌破新低，高点无法到达前一个高点，则不是买点
# 在横盘的低点买入

# 3）最新的10根k线的成交量是最近20根成交量的5倍
# 4）macd在0轴上
import talib.abstract as ta
import pandas as pd
import numpy as np
from freqtrade.strategy.interface import IStrategy

class CustomStrategy0(IStrategy):
    timeframe = '1h'
    stoploss = -0.10
    roi = {
        "0": 0.1,
    }
    trailing_stop = False

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Calculate MACD
        dataframe['macd'], dataframe['signal'], dataframe['hist'] = ta.MACD(dataframe['close'])

        # Calculate the volume ratio of the latest 10 candles to the latest 20 candles
        dataframe['volume_ratio'] = dataframe['volume'].rolling(window=10).sum() / dataframe['volume'].rolling(window=20).sum()

        # Define the criteria for a sideways market (e.g., a price range within a certain percentage)
        sideways_market_pct = 0.08
        dataframe['sideways_market'] = False
        dataframe['sideways_low'] = np.nan

        for i in range(len(dataframe) - 10):
            max_price = dataframe['high'][i:i+10].max()
            min_price = dataframe['low'][i:i+10].min()

            if (max_price - min_price) / min_price <= sideways_market_pct:
                dataframe.at[i + 9, 'sideways_market'] = True
                dataframe.at[i + 9, 'sideways_low'] = min_price

        # Check for continuously declining highs in the sideways market
        dataframe['declining_highs'] = False
        for i in range(len(dataframe) - 20, len(dataframe)):
            if dataframe.at[i, 'sideways_market']:
                declining = True
                for j in range(i - 19, i):
                    if dataframe.at[j, 'high'] >= dataframe.at[j - 1, 'high']:
                        declining = False
                        break
                dataframe.at[i, 'declining_highs'] = declining

        return dataframe

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Define the buy signal conditions
        dataframe.loc[
            (
                (dataframe['sideways_market'] == True) &
                (dataframe['declining_highs'] == False) &
                (dataframe['volume_ratio'] >= 5) &
                (dataframe['macd'] > 0) &
                (dataframe['close'] <= dataframe['sideways_low'])
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Initialize the sell signal column
        dataframe['sell'] = 0

        # Loop through the rows and set the sell signal if the price goes below the previous sideways low
        for i in range(len(dataframe) - 1):
            if dataframe.at[i, 'sideways_market']:
                if dataframe.at[i+1, 'close'] < dataframe.at[i, 'sideways_low']:
                    dataframe.at[i+1, 'sell'] = 1

        return dataframe
