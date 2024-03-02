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


class BigAmount(IStrategy):
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

    # 显示精度位8位小数点
    pd.set_option("display.precision", 8)

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 10
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
    timeframe = '4h'

    # Run "populate_indicators()" only for new candle.
    # 只在新k线才分析，降低CPU
    process_only_new_candles = True

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 66

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


        # 计算大量
        dataframe['volume_ma'] = ta.SMA(dataframe['volume'].values, timeperiod=self.my_short)
        dataframe['volume_rate'] = dataframe['volume'] / dataframe['volume_ma']

        # 成交量大，或者成交量较小且涨幅大
        # 成交量比值、振幅比值>3.0
        dataframe['bigamount'] = ((dataframe['volume_rate'] > self.my_bigamount_volume_rate_small) & (dataframe['delta_rate'] > self.my_bigamount_delta_rate))
        # (dataframe['volume_rate'] > self.my_bigamount_volume_rate)

        # # EMA - Exponential Moving Average
        dataframe['s_ema'] = ta.EMA(dataframe, timeperiod=self.my_short)

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


        dataframe['mybuy'] = (dataframe['bigamount'])


        dataframe.loc[
            (
                (dataframe['mybuy']) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['buy', 'buy_tag']] = (1, 'signal_buy_close_above_s_ma')

        if self.DEBUG:
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

        # 4h sm_ema<0
        dataframe['mysell'] = dataframe['sm_ema'] < 0

        dataframe.loc[
            (
                (dataframe['mysell']) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1

        if self.DEBUG:
            # Print the Analyzed pair
            print(f"result for {metadata['pair']}")

            sell = dataframe[dataframe['sell'] == 1]
            print(sell)

        return dataframe

