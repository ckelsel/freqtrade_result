# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# coding:utf-8 
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from enum import Enum


class test(IStrategy):
    class Scene(Enum):
        UNKNOWN = 1
        BIG_AMOUNT = 2
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_buy_trend, populate_sell_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 1
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.10

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '15m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {
        # Main plot indicators (Moving averages, ...)
        'main_plot': {
            's_ema': {'color': '#7F7F7F'},
            'm_ema': {'color': 'red'},
            'l_ema': {'color': '#4682B4'},
            's_ma': {'color': '#BFBFBF'},
            'm_ma': {'color': '#EE7777'},
            'l_ma': {'color': '#86B0D2'},
        },
        'subplots': {
            # Subplots - each dict defines one additional plot
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }

    DEBUG=False

    # 场景
    scene = Scene.UNKNOWN

    # 双均线周期
    my_short = 8
    my_middle = 33
    my_long = 66

    # 前低，前高
    prev_lower = []
    prev_higher = []

    #
    # 大量
    #
    # volume配置
    my_bigamount_volume_rate = 5.0
    my_bigamount_volume_rate_small = 3.0
    # delta涨幅
    my_bigamount_delta_rate = 0.03

    # 均线密集
    my_dense_delta = 1.0
    my_dense_stoploss_k_line = 96

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    # 多头排列
    def is_full_buy_sequeue(self, cs, sm, ml):
        if cs > 0 and sm > 0 and ml > 0:
            return True
        else:
            return False

    # 多头排列
    def is_sm_ml_buy_sequeue(self, sm, ml):
        if sm > 0 and ml > 0:
            return True
        else:
            return False

    # 空头排列
    def is_full_sell_sequeue(self, cs, sm, ml):
        if cs < 0 and sm < 0 and ml < 0:
            return True
        else:
            return False

    # 前低
    def is_lowest(self, v1, v2, v3):
        if v3 is None:
            return False

        return v2 < v1 and v2 < v3

    # 前低
    def is_prev_lowest(self, v1, v2, v3, v):
        return v2 < v1 and v2 < v3 and v2 < v

    # 前高
    def is_prev_higest(self, v1, v2, v3, v):
        return v2 > v1 and v2 > v3 and v2 > v

    # 寻找收盘价前高
    def search_higher(self, dataframe, maxtrys):
        alen = len(dataframe)
        value = dataframe['high'].values
        if maxtrys > alen:
            maxtrys = alen - 2

        # print("alen {}, maxtrys {}".format(alen, maxtrys))
        first = alen - 1
        for i in range(first, alen - maxtrys, -1):
            if self.is_prev_higest(value[i], value[i-1], value[i-2], value[first]):
                first = i - 1
                break;

        second = first
        for i in range(second, alen - maxtrys, -1):
            if self.is_prev_higest(value[i], value[i-1], value[i-2], value[second]):
                second = i - 1
                break;

        third = second
        for i in range(third, alen - maxtrys, -1):
            if self.is_prev_higest(value[i], value[i-1], value[i-2], value[third]):
                third = i - 1
                break;

        if self.DEBUG:
            print("search_higher: first {}, second {}, third {}".format(first, second, third))
        return [first,second,third]


    # 寻找收盘价前低
    def search_lower(self, dataframe, maxtrys):
        alen = len(dataframe)
        value = dataframe['close'].values
        if maxtrys > alen:
            maxtrys = alen - 2

        # print("alen {}, maxtrys {}".format(alen, maxtrys))
        first = alen - 1
        for i in range(first, alen - maxtrys, -1):
            if self.is_prev_lowest(value[i], value[i-1], value[i-2], value[first]):
                first = i - 1
                break;

        second = first
        for i in range(second, alen - maxtrys, -1):
            if self.is_prev_lowest(value[i], value[i-1], value[i-2], value[second]):
                second = i - 1
                break;

        third = second
        for i in range(third, alen - maxtrys, -1):
            if self.is_prev_lowest(value[i], value[i-1], value[i-2], value[third]):
                third = i - 1
                break;

        if self.DEBUG:
            print("search_lower: first {}, second {}, third {}".format(first, second, third))
        return [first,second,third]

    # 大量
    def search_bigamount(self, dataframe, maxtrys, my_short):
        alen = len(dataframe)
        value = dataframe['volume'].values
        if maxtrys > alen:
            maxtrys = alen

        target = 0
        for i in range(alen - 1, alen - maxtrys, -1):
            subvalue = value[i - my_short - 1:i - 1]
            if value[i] // np.mean(subvalue) >= 5:
                target = i
                break

        return target

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """


        # 计算双均线
        # 
        # # EMA - Exponential Moving Average
        dataframe['s_ema'] = ta.EMA(dataframe, timeperiod=self.my_short)
        dataframe['m_ema'] = ta.EMA(dataframe, timeperiod=self.my_middle)
        dataframe['l_ema'] = ta.EMA(dataframe, timeperiod=self.my_long)

        # # SMA - Simple Moving Average
        dataframe['s_ma'] = ta.SMA(dataframe, timeperiod=self.my_short)
        dataframe['m_ma'] = ta.SMA(dataframe, timeperiod=self.my_middle)
        dataframe['l_ma'] = ta.SMA(dataframe, timeperiod=self.my_long)

        # 计算乖离率
        dataframe.loc[:, 'cs_ma'] = (dataframe['close'] - dataframe['s_ma']) / dataframe['s_ma'] * 100
        dataframe.loc[:, 'sm_ma'] = (dataframe['s_ma'] - dataframe['m_ma']) / dataframe['m_ma'] * 100
        dataframe.loc[:, 'ml_ma'] = (dataframe['m_ma'] - dataframe['l_ma']) / dataframe['l_ma'] * 100

        dataframe.loc[:, 'cs_ema'] = (dataframe['close'] - dataframe['s_ema']) / dataframe['s_ema'] * 100
        dataframe.loc[:, 'sm_ema'] = (dataframe['s_ema'] - dataframe['m_ema']) / dataframe['m_ema'] * 100
        dataframe.loc[:, 'ml_ema'] = (dataframe['m_ema'] - dataframe['l_ema']) / dataframe['l_ema'] * 100

        # 计算振幅
        dataframe['delta'] = dataframe['close'] - dataframe['open']
        dataframe['delta_rate'] = dataframe['delta'] / dataframe['open']

        # 计算多头排列
        dataframe['full_buy_sequeue'] = dataframe.apply(
                lambda row: self.is_full_buy_sequeue(row['cs_ma'], row['sm_ma'], row['ml_ma']),
                axis=1, raw=False)

        dataframe['sm_ml_buy_sequeue'] = dataframe.apply(
                lambda row: self.is_sm_ml_buy_sequeue(row['sm_ma'], row['ml_ma']),
                axis=1, raw=False)

        dataframe['ml_buy_sequeue'] = dataframe['ml_ma'] > 0

        # 计算空头排列
        dataframe['full_sell_sequeue'] = dataframe.apply(
                lambda row: self.is_full_sell_sequeue(row['cs_ma'], row['sm_ma'], row['ml_ma']),
                axis=1, raw=False)

        # 计算大量
        dataframe['volume_ma'] = ta.SMA(dataframe['volume'].values, timeperiod=self.my_short)
        dataframe['volume_rate'] = dataframe['volume'] / dataframe['volume_ma']

        # 成交量大，或者成交量较小且涨幅大
        # 成交量比值、振幅比值>3.0
        dataframe['bigamount'] = ((dataframe['volume_rate'] > self.my_bigamount_volume_rate_small) & (dataframe['delta_rate'] > self.my_bigamount_delta_rate))
        # (dataframe['volume_rate'] > self.my_bigamount_volume_rate)

        # 计算均线密集
        # cs, sm, ml < 1.0
        dataframe['dense_ema'] = (abs(dataframe['ml_ema']) < self.my_dense_delta) & \
            (abs(dataframe['ml_ma']) < self.my_dense_delta) & \
            (abs(dataframe['cs_ema']) < self.my_dense_delta) & \
            (abs(dataframe['cs_ma']) < self.my_dense_delta) & \
            (abs(dataframe['sm_ema']) < self.my_dense_delta) & \
            (abs(dataframe['sm_ma']) < self.my_dense_delta)

        # 识别低点
        # n-1,n下跌，n+1上涨
        # close: n-1>n, n < n+1
        dataframe['is_lowest'] = False
        dataframe['is_lowest'] = (dataframe['close'] < dataframe['close'].shift(periods=1, fill_value=0.0)) & \
                (dataframe['close'] < dataframe['close'].shift(periods=-1, fill_value=0.0)) & \
                (dataframe['delta'] < 0) & \
                (dataframe['delta'].shift(periods=1, fill_value=-1.0) < 0) & \
                (dataframe['delta'].shift(periods=-1, fill_value=-1.0) > 0)

        # 识别高点
        # n-1,n上涨，n+1下跌
        # high: n-1<n, n > n+1
        # 前一个:period=1
        dataframe['is_highest'] = False
        dataframe['is_highest'] = (dataframe['close'] > dataframe['close'].shift(periods=1, fill_value=0.0)) & \
                (dataframe['close'] > dataframe['close'].shift(periods=-1, fill_value=0.0)) & \
                (dataframe['delta'] > 0) & \
                (dataframe['delta'].shift(periods=1, fill_value=0.0) > 0) & \
                (dataframe['delta'].shift(periods=-1, fill_value=0.0) < 0)


        # 寻找三个前低
        self.prev_lower = self.search_lower(dataframe, self.my_middle)
        dataframe['prev_lower'] = False
        dataframe.loc[self.prev_lower[0], 'prev_lower'] = True
        dataframe.loc[self.prev_lower[1], 'prev_lower'] = True
        dataframe.loc[self.prev_lower[2], 'prev_lower'] = True
        # print(self.prev_lower)
        # print(dataframe.iloc[self.prev_lower[0]])
        # print(dataframe.iloc[self.prev_lower[1]])
        # print(dataframe.iloc[self.prev_lower[2]])

        # 寻找三个前高
        self.prev_higher = self.search_higher(dataframe, self.my_middle)
        dataframe['prev_higher'] = False
        dataframe.loc[self.prev_higher[0], 'prev_higher'] = True
        dataframe.loc[self.prev_higher[1], 'prev_higher'] = True
        dataframe.loc[self.prev_higher[2], 'prev_higher'] = True
        # print(dataframe.iloc[self.prev_higher[0]])
        # print(dataframe.iloc[self.prev_higher[1]])
        # print(dataframe.iloc[self.prev_higher[2]])


        if self.DEBUG:
            print(dataframe.tail(40))

        """
        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        """

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        #
        # 大量
        #

        # 多头排列
        dataframe['buy_bigamount'] = (dataframe['ml_buy_sequeue'] > 0) & (dataframe['bigamount'])

        #
        # 均线密集
        #

        dataframe['buy_dense_ma'] = dataframe['dense_ema'] & dataframe['is_lowest']

        dataframe.loc[
            (
                (dataframe['buy_dense_ma']) &
                #(dataframe['buy_bigamount']) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['buy', 'buy_tag']] = (1, 'buy_signal_bigamount')

        # Print the Analyzed pair
        print(f"result for {metadata['pair']}")

        buy = dataframe[dataframe['buy'] == 1]
        print(buy)

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        #
        # 大量
        #

        # 真对大量的买点buy，设置大量低点止损
        bigamount_map = {}
        dataframe['sell_bigamount'] = False
        bigamount = dataframe[dataframe['buy_bigamount']]
        bigamount_length = len(bigamount)
        for index, row in bigamount.iterrows():
            df = dataframe[index:]
            # close低于大量低点low
            target = df[df.close < row['low']].first_valid_index()
            #print("target {}, index {}".format(target, index))
            bigamount_map[index] = target
            if target is not None:
                dataframe.loc[target, 'sell_bigamount'] = True

        # TODO
        # 连续大量，删除前一个大量的卖点


        #
        # 均线密集
        #

        # 跌破前低，卖出
        dataframe['sell_dense_ma'] = False
        prev_lower = self.search_lower(dataframe, self.my_dense_stoploss_k_line)
        stoploss_close = dataframe.loc[prev_lower[2], 'close']
        dataframe['sell_dense_ma'] = dataframe['close'] < stoploss_close
        print("stoploss_close {}".format(stoploss_close))

        dataframe.loc[
            (
                (dataframe['sell_dense_ma']) &
                #(dataframe['sell_bigamount']) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1

        # Print the Analyzed pair
        print(f"result for {metadata['pair']}")

        sell = dataframe[dataframe['sell'] == 1]
        print(sell)

        return dataframe

