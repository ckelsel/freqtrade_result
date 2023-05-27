# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# coding:utf-8 
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
import logging

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from enum import Enum


logger = logging.getLogger(__name__)

class ma_roi5(IStrategy):
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
        "0": 0.011
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.1

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '15m'

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
    my_short = 20
    my_middle = 60
    my_long = 120

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

        # logger.info("alen {}, maxtrys {}".format(alen, maxtrys))
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
            logger.info("search_higher: first {}, second {}, third {}".format(first, second, third))
        return [first,second,third]


    # 寻找收盘价前低
    def search_lower(self, dataframe, maxtrys):
        alen = len(dataframe)
        value = dataframe['close'].values
        if maxtrys > alen:
            maxtrys = alen - 2

        # logger.info("alen {}, maxtrys {}".format(alen, maxtrys))
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
            logger.info("search_lower: first {}, second {}, third {}".format(first, second, third))
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

        # 只处理120个k线,加快处理速度
        # TODO: interface.py  analyze_pair会比对分析前后的dataframe
        #  len1=len(dataframe)
        #  dataframe = dataframe[len1-120:]

        #  #
        #  # 多周期
        #  #
        #  ohlc = {
        #      'open': 'first',
        #      'high': 'max',
        #      'low': 'min',
        #      'close': 'last',
        #      'volume': 'sum'
        #  }
        #
        #  #  print(dataframe)
        #  dataframe = dataframe.set_index('date')
        #  df60 = dataframe.resample('60min').apply(ohlc)
        #  # 可能出现Nan
        #  df60.fillna(method='ffill', inplace=True)
        #
        #  # 计算双均线
        #  #
        #  # # EMA - Exponential Moving Average
        #  df60.loc[:, 's_ema60'] = ta.EMA(df60.close, timeperiod=self.my_short)
        #  df60.loc[:, 'm_ema60'] = ta.EMA(df60.close, timeperiod=self.my_middle)
        #  df60.loc[:, 'l_ema60'] = ta.EMA(df60.close, timeperiod=self.my_long)
        #
        #  # # SMA - Simple Moving Average
        #  df60.loc[:, 's_ma60'] = ta.SMA(df60.close, timeperiod=self.my_short)
        #  df60.loc[:, 'm_ma60'] = ta.SMA(df60.close, timeperiod=self.my_middle)
        #  df60.loc[:, 'l_ma60'] = ta.SMA(df60.close, timeperiod=self.my_long)
        #
        #  # 计算乖离率
        #  df60.loc[:, 'cs_ma60'] = (df60['close'] - df60['s_ma60']) / df60['s_ma60'] * 100
        #  df60.loc[:, 'sm_ma60'] = (df60['s_ma60'] - df60['m_ma60']) / df60['m_ma60'] * 100
        #  df60.loc[:, 'ml_ma60'] = (df60['m_ma60'] - df60['l_ma60']) / df60['l_ma60'] * 100
        #
        #  df60.loc[:, 'cs_ema60'] = (df60['close'] - df60['s_ema60']) / df60['s_ema60'] * 100
        #  df60.loc[:, 'sm_ema60'] = (df60['s_ema60'] - df60['m_ema60']) / df60['m_ema60'] * 100
        #  df60.loc[:, 'ml_ema60'] = (df60['m_ema60'] - df60['l_ema60']) / df60['l_ema60'] * 100
        #
        #  #  print(df60.to_string())
        #
        #  # 复制
        #  dataframe.loc[:,'s_ema60'] = df60['s_ema60']
        #  dataframe['s_ema60'].fillna(method='ffill', inplace=True)
        #  dataframe.loc[:,'m_ema60'] = df60['m_ema60']
        #  dataframe['m_ema60'].fillna(method='ffill', inplace=True)
        #  dataframe.loc[:,'l_ema60'] = df60['l_ema60']
        #  dataframe['l_ema60'].fillna(method='ffill', inplace=True)
        #
        #  dataframe.loc[:,'s_ma60'] = df60['s_ma60']
        #  dataframe['s_ma60'].fillna(method='ffill', inplace=True)
        #  dataframe.loc[:,'m_ma60'] = df60['m_ma60']
        #  dataframe['m_ma60'].fillna(method='ffill', inplace=True)
        #  dataframe.loc[:,'l_ma60'] = df60['l_ma60']
        #  dataframe['l_ma60'].fillna(method='ffill', inplace=True)
        #
        #  dataframe.loc[:,'cs_ma60'] = df60['cs_ma60']
        #  dataframe['cs_ma60'].fillna(method='ffill', inplace=True)
        #  dataframe.loc[:,'sm_ma60'] = df60['sm_ma60']
        #  dataframe['sm_ma60'].fillna(method='ffill', inplace=True)
        #  dataframe.loc[:,'ml_ma60'] = df60['ml_ma60']
        #  dataframe['ml_ma60'].fillna(method='ffill', inplace=True)
        #
        #  dataframe.loc[:,'cs_ema60'] = df60['cs_ema60']
        #  dataframe['cs_ema60'].fillna(method='ffill', inplace=True)
        #  dataframe.loc[:,'sm_ema60'] = df60['sm_ema60']
        #  dataframe['sm_ema60'].fillna(method='ffill', inplace=True)
        #  dataframe.loc[:,'ml_ema60'] = df60['ml_ema60']
        #  dataframe['ml_ema60'].fillna(method='ffill', inplace=True)
        #
        #  #  print(dataframe.to_string())
        #  dataframe = dataframe.reset_index()
        #
        #  # 计算双均线
        #  #
        #  # # EMA - Exponential Moving Average
        #  dataframe.loc[:, 's_ema'] = ta.EMA(dataframe, timeperiod=self.my_short)
        #  dataframe.loc[:, 'm_ema'] = ta.EMA(dataframe, timeperiod=self.my_middle)
        #  dataframe.loc[:, 'l_ema'] = ta.EMA(dataframe, timeperiod=self.my_long)
        #
        # # SMA - Simple Moving Average
        dataframe.loc[:, 's_ma'] = ta.SMA(dataframe, timeperiod=self.my_short)
        dataframe.loc[:, 'm_ma'] = ta.SMA(dataframe, timeperiod=self.my_middle)
        dataframe.loc[:, 'l_ma'] = ta.SMA(dataframe, timeperiod=self.my_long)

        # 计算乖离率
        dataframe.loc[:, 'cs_ma'] = (dataframe['close'] - dataframe['s_ma']) / dataframe['s_ma'] * 100
        dataframe.loc[:, 'sm_ma'] = (dataframe['s_ma'] - dataframe['m_ma']) / dataframe['m_ma'] * 100
        dataframe.loc[:, 'ml_ma'] = (dataframe['m_ma'] - dataframe['l_ma']) / dataframe['l_ma'] * 100

        # 计算波动率
        TRADING_DAYS = 252
        returns = np.log(dataframe['close']/dataframe['close'].shift(1))
        returns.fillna(0, inplace=True)
        volatility = returns.rolling(window=TRADING_DAYS).std()*np.sqrt(TRADING_DAYS)
        dataframe['volatility'] = volatility.values
        #
        #  dataframe.loc[:, 'cs_ema'] = (dataframe['close'] - dataframe['s_ema']) / dataframe['s_ema'] * 100
        #  dataframe.loc[:, 'sm_ema'] = (dataframe['s_ema'] - dataframe['m_ema']) / dataframe['m_ema'] * 100
        #  dataframe.loc[:, 'ml_ema'] = (dataframe['m_ema'] - dataframe['l_ema']) / dataframe['l_ema'] * 100
        #
        #  dataframe.loc[:, 'ml_ema_s_deduction'] = dataframe['ml_ema'].shift(periods=self.my_short, fill_value=0.0)
        #
        #  # 计算抵扣价
        #  dataframe.loc[:, 's_deduction'] = dataframe['close'].shift(periods=self.my_short, fill_value=0.0)
        #  dataframe.loc[:, 's_deduction_rate'] = (dataframe['close'] - dataframe['s_deduction']) / dataframe['close']
        #
        #  dataframe.loc[:, 'm_deduction'] = dataframe['close'].shift(periods=self.my_middle, fill_value=0.0)
        #  dataframe.loc[:, 'm_deduction_rate'] = (dataframe['close'] - dataframe['s_deduction']) / dataframe['close']
        #
        #  dataframe.loc[:, 'l_deduction'] = dataframe['close'].shift(periods=self.my_long, fill_value=0.0)
        #  dataframe.loc[:, 'l_deduction_rate'] = (dataframe['close'] - dataframe['s_deduction']) / dataframe['close']
        #
        #  #  print(dataframe[['date', 'close', 's_deduction', 's_deduction_rate', 'm_deduction', 'm_deduction_rate', 'l_deduction', 'l_deduction_rate']].head(80).to_string())
        #
        #
        #  #  # close收在ma8
        #  dataframe.loc[:, 'close_above_s_ma'] = dataframe['close'] > dataframe['s_ma']
        #  #
        #  # 计算振幅
        dataframe.loc[:, 'delta'] = dataframe['close'] - dataframe['open']
        #  dataframe.loc[:, 'delta_rate'] = dataframe['delta'] / dataframe['open']
        #
        #  # 计算多头排列
        #  dataframe.loc[:, 'full_buy_sequeue'] = dataframe.apply(
        #          lambda row: self.is_full_buy_sequeue(row['cs_ema'], row['sm_ema'], row['ml_ema']),
        #          axis=1, raw=False)
        #
        #  dataframe.loc[:, 'sm_ml_buy_sequeue'] = dataframe.apply(
        #          lambda row: self.is_sm_ml_buy_sequeue(row['sm_ma'], row['ml_ma']),
        #          axis=1, raw=False)
        #
        #  dataframe.loc[:, 'ml_buy_sequeue'] = dataframe['ml_ma'] > 0
        #
        #  # 计算空头排列
        dataframe.loc[:, 'full_sell_sequeue'] = dataframe.apply(
                lambda row: self.is_full_sell_sequeue(row['cs_ma'], row['sm_ma'], row['ml_ma']),
                axis=1, raw=False)
        #
        #  # 计算大量
        #  dataframe.loc[:, 'volume_ma'] = ta.SMA(dataframe['volume'].values, timeperiod=self.my_short)
        #  dataframe.loc[:, 'volume_rate'] = dataframe['volume'] / dataframe['volume_ma']
        #
        #  成交量大，或者成交量较小且涨幅大
        #  成交量比值、振幅比值>3.0
        #  dataframe.loc[:, 'bigamount'] = ((dataframe['volume_rate'] > self.my_bigamount_volume_rate_small) & (dataframe['delta_rate'] > self.my_bigamount_delta_rate))
        #  # (dataframe['volume_rate'] > self.my_bigamount_volume_rate)
        #
        #  # 计算均线密集
        #  # cs, sm, ml < 1.0
        #  dataframe.loc[:, 'dense_ema'] = (abs(dataframe['ml_ema']) < self.my_dense_delta) & \
        #      (abs(dataframe['ml_ma']) < self.my_dense_delta) & \
        #      (abs(dataframe['cs_ema']) < self.my_dense_delta) & \
        #      (abs(dataframe['cs_ma']) < self.my_dense_delta) & \
        #      (abs(dataframe['sm_ema']) < self.my_dense_delta) & \
        #      (abs(dataframe['sm_ma']) < self.my_dense_delta)

        #
        # 识别低点
        #

        # n-1,n下跌，n+1上涨
        # close: n-1>n, n < n+1
        dataframe.loc[:, 'is_lowest'] = False
        dataframe.loc[:, 'is_lowest'] = (dataframe['close'] < dataframe['close'].shift(periods=1, fill_value=0.0)) & \
                (dataframe['close'] < dataframe['close'].shift(periods=-1, fill_value=0.0)) & \
                (dataframe['delta'] < 0)

        # 前低
        dataframe.loc[:, 'prev_low'] = dataframe.query('is_lowest==True')['close']


        # 前前低
        dataframe.loc[:, 'prev2_low'] = None
        prev_low = None
        for index, row in dataframe.iterrows():
            if not np.isnan(row['prev_low']):
                if prev_low is None:
                    prev_low = row['prev_low']
                else:
                    dataframe.loc[index, 'prev2_low'] = prev_low
                    prev_low = row['prev_low']


        dataframe['prev_low'].fillna(method='ffill', inplace=True)
        dataframe['prev2_low'].fillna(method='ffill', inplace=True)

        #
        # 识别高点
        #

        # n-1,n上涨，n+1下跌
        # high: n-1<n, n > n+1
        # 前一个:period=1
        dataframe.loc[:, 'is_highest'] = False
        dataframe.loc[:, 'is_highest'] = (dataframe['close'] > dataframe['close'].shift(periods=1, fill_value=0.0)) & \
                (dataframe['close'] > dataframe['close'].shift(periods=-1, fill_value=0.0)) & \
                (dataframe['delta'] > 0)

        # 前高
        dataframe.loc[:, 'prev_high'] = dataframe.query('is_highest==True')['close']


        # 前前高
        dataframe.loc[:, 'prev2_high'] = None
        prev_high = None
        for index, row in dataframe.iterrows():
            if not np.isnan(row['prev_high']):
                if prev_high is None:
                    prev_high = row['prev_high']
                else:
                    dataframe.loc[index, 'prev2_high'] = prev_high
                    prev_high = row['prev_high']


        dataframe['prev_high'].fillna(method='ffill', inplace=True)
        dataframe['prev2_high'].fillna(method='ffill', inplace=True)


        dataframe = self.get_prev_is_lowest(dataframe)

        #  if self.DEBUG:
        #      print(dataframe.tail(40).to_string())

        #  debug_columns=['date','open', 'high', 'low', 'close', 'is_highest', 'prev_high', 'prev2_high', 'is_lowest', 'prev_low', 'prev2_low']
        #  print(dataframe[debug_columns].tail(40).to_string())

        """
        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        """

        return dataframe


    def get_prev_is_lowest(self, data):
        data[["date"]] = data[["date"]].astype(str)
        data['row_num'] = data.date.rank(method='min').astype(int)
        data_copy = data.copy()
        data_copy.row_num = data_copy.row_num.apply(lambda x: x + 1)

        data_copy.rename(columns={'is_lowest': 'prev_is_lowest' },
                     inplace=True)
        data_copy = data_copy[['row_num', 'prev_is_lowest'
                           ]]
        data = data.set_index(['row_num'])
        data_copy = data_copy.set_index(['row_num'])
        data = pd.merge(data, data_copy, how='left', on=['row_num'])
        data = data.reset_index()
        data['date'] = pd.to_datetime(data['date'])
        return data

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        #  # 均线密集的启动
        #  # 均线密集
        #  # 下跌
        #  # 上一个周期不是下跌
        #  dataframe.loc[:, 'mybuy1'] = (dataframe['dense_ema']) & \
        #          (dataframe['delta'] < 0)
        #          #  (abs(dataframe['close'] / dataframe['m_ma']) < 1.005)
        #
        #  # 上涨过程的回调
        #  # 长期均线多头
        #  # 中期均线多头
        #  # close 跌破短期均线
        #  # 最低价
        #  dataframe.loc[:, 'mybuy2'] = (dataframe['m_ema'] < 1.0) & \
        #          (dataframe['l_ema'] < 0.5) & \
        #          (dataframe['close'] < dataframe['s_ma']) & \
        #          (dataframe['sm_ml_buy_sequeue']) & \
        #          (dataframe['prev_is_lowest'])
        #
        #
        #  # 抵扣价
        #  dataframe.loc[:, 'mybuy3'] = \
        #          (dataframe['close'] / dataframe['s_deduction'] > 1.02) & \
        #          (dataframe['cs_ema'] > 0.0) & \
        #          (dataframe['cs_ma'] > 0.0) & \
        #          (dataframe['sm_ema'] > 0.0) & \
        #          (dataframe['sm_ma'] > 0.0) & \
        #          (dataframe['ml_ema'] > 0.0) & \
        #          (dataframe['ml_ma'] > 0.0) & \
        #          (dataframe['bigamount'])
        #
        #  # 前低
        #  # 抵扣价>中期抵扣价高
        #  # 60min的cs, sm > 0
        #  #  dataframe.loc[:, 'mybuy4'] = (dataframe['prev_is_lowest']) & \
        #  #          (dataframe['close'] / dataframe['m_deduction'] > 1.02) & \
        #  #          (dataframe['cs_ema60'] > 0) & \
        #  #          (dataframe['ml_ema60'] > 0) & \
        #  #          (dataframe['sm_ema60'] > 0)
        #
        #  # 短，中，长抵扣价
        #  dataframe.loc[:, 'mybuy5'] = \
        #          (dataframe['close'] / dataframe['s_deduction'] > 1.01) & \
        #          (dataframe['s_deduction'] / dataframe['m_deduction'] > 1.01) & \
        #          (dataframe['m_deduction'] / dataframe['l_deduction'] > 1.01)
        #
        #  # 乖离率不能大，sm < ml
        #  # ml > 0.7，多头排列
        #  # ml 逐渐变大
        #  # 中期ml > 0, 不是从下跌趋势，马上转上涨
        #  # close大于中期抵扣价
        #  # 只买跌
        #  # 跨周期 ml > 0，更大的周期不是下跌趋势
        #  #  dataframe.loc[:, 'mybuy6'] = \
        #  #          ((dataframe['sm_ema'] < dataframe['ml_ema']) & (dataframe['sm_ma'] < dataframe['ml_ma'])) & \
        #  #          ((dataframe['ml_ema'] > 0.7) & (dataframe['ml_ma'] > 0.7)) & \
        #  #          (dataframe['ml_ema'] > dataframe['ml_ema_s_deduction']) & \
        #  #          (dataframe['ml_ema_s_deduction'] > 0) & \
        #  #          (dataframe['close'] > dataframe['m_deduction']) & \
        #  #          (dataframe['delta'] < 0) & \
        #  #          (dataframe['ml_ma60'] > 0)
        #
        #          #  ((dataframe['sm_ema'] < 1.0) & \
        #          #  (dataframe['sm_ma'] < 1.0) & \
        #          #  (dataframe['ml_ema'] > 1.5) & \
        #          #  (dataframe['ml_ma'] > 1.5)) & \
        #
        #  # 在前一个最低价买入，且没有卖出信号，只使用止损
        #  dataframe.loc[:, 'mybuy7'] = (dataframe['prev_is_lowest'])
        #
        #  # 捕捉上升趋势
        #  # 上涨
        #  # close > 短、中、长期抵扣价
        #  # close站上s_ma
        #  dataframe.loc[:, 'mybuy8'] = \
        #          (dataframe['delta'] > 0) & \
        #          (dataframe['close'] / dataframe['s_deduction'] > 1.005) & \
        #          (dataframe['close'] > dataframe['s_deduction']) & \
        #          (dataframe['close'] > dataframe['m_deduction']) & \
        #          (dataframe['close'] > dataframe['l_deduction']) & \
        #          (dataframe['close'] > dataframe['s_ma'])


        # 前高比前前高高
        # 前低比前前低高
        # 不是完全空头排列
        # 不能在最高价、最低价成交，因为实盘不能这么做
        # 波动率大于0.1
        dataframe.loc[:, 'mybuy9'] = \
                (dataframe['prev_high'] > dataframe['prev2_high']) & \
                (dataframe['prev_low'] > dataframe['prev2_high']) & \
                (dataframe['full_sell_sequeue'] == False) & \
                (dataframe['is_highest'] == False) & \
                (dataframe['is_lowest'] == False) & \
                (dataframe['volatility'] > 0.1) & \
                (dataframe['delta'] > 0)

        dataframe.loc[:, 'mybuy'] = dataframe['mybuy9']
        ##dataframe.loc[:, 'mybuy'] = dataframe['mybuy1']

        # 4h sm_ema>0
        #dataframe.loc[:, 'mybuy'] = dataframe['sm_ema'] > 0

        dataframe.loc[
            (
                (dataframe['mybuy']) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['buy', 'buy_tag']] = (1, 'signal_buy_close_above_s_ma')

        if self.DEBUG:
            #print(f"populate_buy_trend len {len(dataframe)}")

            # Print the Analyzed pair
            buy = dataframe[dataframe['buy'] == 1]
            buy_today=buy[buy['row_num'] == dataframe.tail(1).row_num.values[0]]
            #  logger.info(f"buy result for {metadata['pair']}")
            #  logger.info(dataframe.tail(1).to_string())
            #  logger.info(buy_today)
            #  logger.info(len(buy_today))
            if len(buy_today) != 0:
                logger.info(f"buy result for {metadata['pair']}")
                logger.info(buy_today.to_string())
                logger.info(buy.tail(5).to_string())
                logger.info(dataframe.tail(5).to_string())

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        #  # 4h sm_ema<0
        #  #dataframe.loc[:, 'mysell'] = dataframe['sm_ema'] < 0
        #
        #  # 空头排列
        #  dataframe.loc[:, 'mysell1'] = dataframe['full_sell_sequeue'] & (abs(dataframe['cs_ema']) > 1.0)
        #
        #  # 大幅回撤
        #  #  dataframe.loc[:, 'mysell2'] = (dataframe['cs_ema'] < 0) & (dataframe['sm_ema'] < 0) & (dataframe['ml_ema'] > 0)
        #
        #  # 单笔6个点
        #  #dataframe.loc[:, 'mysell'] = dataframe['full_sell_sequeue'] | ((abs(dataframe['delta_rate']) > 0.06) & dataframe['full_buy_sequeue'])
        #
        #  # 空头排列
        #  # cs, ml < 0
        #  #dataframe.loc[:, 'mysell'] = ((dataframe['cs_ema'] < 0) & (dataframe['ml_ema'] < 0)) | dataframe['full_sell_sequeue']
        #
        #  # 抵扣价跟当前价格一样
        #  dataframe.loc[:, 'mysell3'] = (dataframe['s_deduction'] / dataframe['close'] > 1.02) | \
        #          (dataframe['sm_ema'] < 0) | \
        #          (dataframe['sm_ma'] < 0)
        #
        #
        #  #  dataframe.loc[:, 'mysell4'] = \
        #  #          (dataframe['ml_ema'] < dataframe['ml_ema_s_deduction'])
        #
        #  # 均线密集
        #  dataframe.loc[:, 'mysell5'] = (dataframe['dense_ema']) & \
        #          (dataframe['close'] < dataframe['m_ma']) & \
        #          (dataframe['delta'] > 0) & \
        #          (dataframe['ml_ema'] > 0) & \
        #          (dataframe['ml_ma'] > 0)
        #

        # 前低低于前前低
        # 前高低于前前高
        # 下跌close低于前低
        # 不能在最高价、最低价成交，因为实盘不能这么做
        dataframe.loc[:, 'mysell5'] = \
                (dataframe['prev_low'] < dataframe['prev2_low']) & \
                (dataframe['prev_high'] < dataframe['prev2_high']) & \
                (dataframe['is_highest'] == False) & \
                (dataframe['is_lowest'] == False)

        dataframe.loc[:, 'mysell'] = (dataframe['mysell5'])

        dataframe.loc[
            (
                (dataframe['mysell']) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1

        if self.DEBUG:
            #print(f"populate_sell_trend len {len(dataframe)}")

            #  77 Columns: [row_num, date, open, high, low, close, volume, s_ma, m_ma, l_ma, cs_ma, sm_ma, ml_ma, volatility, delta, full_sell_sequeue, is_lowest, prev_low, prev2_low, is_highest, prev_high, prev2_high, pr>
            # Print the Analyzed pair
            sell = dataframe[dataframe['sell'] == 1]
            sell_today=sell[sell['row_num'] == dataframe.tail(1).row_num.values[0]]
            #  logger.info(f"sell result for {metadata['pair']}")
            #  logger.info(sell['row_num'])
            #  logger.info(dataframe.tail(1).row_num.values[0])
            #  logger.info(dataframe.tail(1).to_string())
            #  logger.info(sell_today)
            #  logger.info(len(sell_today))
            if len(sell_today) != 0:
                logger.info(f"sell result for {metadata['pair']}")
                logger.info(sell_today.to_string())
                logger.info(sell.tail(5).to_string())
                logger.info(dataframe.tail(5).to_string())

        return dataframe

