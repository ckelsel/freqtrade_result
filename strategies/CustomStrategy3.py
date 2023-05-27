# 在原始横盘策略的基础上，我们可以通过以下方法进行优化：

# 添加交易量指标：我们可以计算每个K线的平均交易量，并在横盘市场中寻找较低的交易量。

# 使用其他技术指标：我们可以添加布林带指标，以确保价格在横盘期间保持在布林带的中间范围内。我们还可以添加RSI指标，以确保横盘期间的RSI值接近中性（例如，接近50）。

# 使用多个时间周期：为了在更长的时间范围内检查横盘市场，我们可以在1小时和4小时时间周期上进行分析。

# 考虑市场情绪：虽然在策略代码中直接添加市场情绪分析可能较为复杂，但您可以在实际操作时将其视为一个参考因素。关注新闻和社交媒体上的市场情绪，并在决策时考虑这些信息。
import talib.abstract as ta
import pandas as pd
import numpy as np
from freqtrade.strategy.interface import IStrategy

class CustomStrategy3(IStrategy):
    timeframe = '1h'
    stoploss = -0.10
    roi = {
        "0": 0.1,
    }
    trailing_stop = False

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 计算 MACD
        dataframe['macd'], dataframe['signal'], dataframe['hist'] = ta.MACD(dataframe['close'])

        # 计算布林带
        dataframe['bb_lowerband'], dataframe['bb_middleband'], dataframe['bb_upperband'] = ta.BBANDS(dataframe['close'])

        # 计算 RSI
        dataframe['rsi'] = ta.RSI(dataframe['close'])

        # 定义横盘市场的条件
        sideways_market_pct = 0.08
        dataframe['sideways_market'] = False
        dataframe['sideways_start'] = np.nan

        for i in range(len(dataframe) - 12):
            max_price = dataframe['high'][i:i+12].max()
            min_price = dataframe['low'][i:i+12].min()
            price_amplitude = (max_price - min_price) / min_price

            if price_amplitude <= sideways_market_pct:
                # 布林带宽度
                bb_width = (dataframe.at[i + 11, 'bb_upperband'] - dataframe.at[i + 11, 'bb_lowerband']) / dataframe.at[i + 11, 'bb_middleband']
                # RSI 处于中性区域
                rsi_neutral = 30 < dataframe.at[i + 11, 'rsi'] < 70

                if bb_width <= 0.1 and rsi_neutral:
                    dataframe.at[i + 11, 'sideways_market'] = True
                    dataframe.at[i + 11, 'sideways_start'] = i

        return dataframe

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 定义买入信号条件
        dataframe.loc[
            (
                dataframe['sideways_market'] &
                (dataframe['macd'] > 0)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 初始化卖出信号列
        dataframe['sell'] = 0

        # 遍历行并设置卖出信号，如果价格低于前一个横盘的最低点
        for i in range(len(dataframe) - 1):
            if dataframe.at[i, 'sideways_market']:
                if dataframe.at[i+1, 'close'] < dataframe.at[i, 'sideways_low']:
                    dataframe.at[i+1, 'sell'] = 1

        return dataframe
