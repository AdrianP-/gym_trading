import numpy as np
import pandas as pd
import talib
import time


class FeatureEngineering(object):
    def __init__(self, df: pd.DataFrame):

        df = self._calculate_return(df)
        df = self._generate_indicators(df)
        # df = self._price_engineering(df)
        # df = self._volume_engineering(df)

        df['Open Trade'] = np.zeros(df.shape[0])
        df['Duration Trade'] = np.zeros(df.shape[0])

        df = df.dropna()

        self.df = df

    def get_df_processed(self):
        return self.df

    def _calculate_return(self, df):
        # Compute Returns based on consecutive closed prices
        df['Return'] = df['Close'].pct_change() * 100

        # Normalization of returns
        mean = df['Return'].mean()
        std = df['Return'].std()
        df['Return'] = (df['Return'] - np.array(mean)) / np.array(std)

        return df

    def _generate_indicators(self, data):
        _high = data.High.values
        _low = data.Low.values
        _close = data.Close.values

        # Compute the ATR and perform Normalisation
        data['ATR'] = talib.ATR(_high, _low, _close, timeperiod=14)
        data.dropna(inplace=True)
        data['ATR'] = (data['ATR'] - np.mean(data['ATR'])) / np.std(data['ATR'])

        return data

    def _price_engineering(self, df):
        # Price Engineering
        # Get opens
        period_list = [1, 2, 3, 4, 5, 10, 21, 63]
        for x in period_list:
            df['-' + str(x) + 'd_Open'] = df['Open'].shift(x)

        # # Get adjCloses
        # period_list = range(1, 5 + 1)
        # for x in period_list:
        #     df['-' + str(x) + 'd_adjClose'] = df['Adj Close'].shift(x)

        #Get closes
        # Get adjCloses
        period_list = range(1, 5 + 1)
        for x in period_list:
            df['-' + str(x) + 'd_Close'] = df['Close'].shift(x)

        # Get highs
        period_list1 = range(1, 5 + 1)
        for x in period_list1:
            df['-' + str(x) + 'd_High'] = df['High'].shift(x)

        period_list2 = [10, 21, 63, 100]
        for x in period_list2:
            df[str(x) + 'd_High'] = df['High'].shift().rolling(window=x).max()

        # Get lows
        period_list1 = range(1, 5 + 1)
        for x in period_list1:
            df['-' + str(x) + 'd_Low'] = df['Low'].shift(x)

        period_list2 = [10, 21, 63, 100]
        for x in period_list2:
            df[str(x) + 'd_Low'] = df['High'].shift().rolling(window=x).min()

        return df

    def _all_candlestick_patterns(self, df):
        h = df.High.values
        o = df.Open.values
        l = df.Low.values
        c = df.Close.values
        v = df.Volume.values

        name = 'CDL2CROWS'
        df[name] = talib.CDL2CROWS(o,h,l,c)
        talib.get_function_groups()
        return df

