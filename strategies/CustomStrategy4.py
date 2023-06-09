# 横盘过程中，突然下跌，再次拉升，形成双顶，接着大跌，这时候需要卖出

import talib.abstract as ta
import pandas as pd
import numpy as np
from freqtrade.strategy.interface import IStrategy

import talib.abstract as ta
import pandas as pd
import numpy as np
from freqtrade.strategy.interface import IStrategy

class CustomStrategy4(IStrategy):
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

        # Add indicators for double top pattern detection
        dataframe['doji'] = ta.CDLDOJI(dataframe)


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

        # Loop through the rows and set the sell signal if a double top pattern is detected
        for i in range(len(dataframe) - 1):
            if dataframe.at[i, 'doji'] and dataframe.at[i - 1, 'doji']:
                dataframe.at[i+1, 'sell'] = 1


        # 遍历行并设置卖出信号，如果价格低于前一个横盘的最低点
        for i in range(len(dataframe) - 1):
            if dataframe.at[i, 'sideways_market']:
                if dataframe.at[i+1, 'close'] < dataframe.at[i, 'sideways_low']:
                    dataframe.at[i+1, 'sell'] = 1

        return dataframe

