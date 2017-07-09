import numpy as np
import pandas as pd
from .feature_engineering import FeatureEngineering


class Simulator(object):
    def __init__(self, csv_name, train_split, dummy_period=None, train=True, multiple_trades=False):
        if "EUR" in csv_name:
            df = pd.read_csv(csv_name, parse_dates=[[0, 1]], header=None,
                             names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df = df[~np.isnan(df['Open'])].set_index('Date_Time')

        else:
            df = pd.read_csv(csv_name, usecols=['Date', 'High', 'Low', 'Open', 'Close', 'Volume'])
            df = df[~np.isnan(df['Open'])].set_index('Date')

        df = FeatureEngineering(df).get_df_processed()

        ##Attributes
        self.data = df
        self.date_time = df.index
        self.count = df.shape[0]
        self.train_end_index = int(train_split * self.count)

        # Attributes related to the observation state: Return
        # print(self.data.head(1))

        data_dropped = self.data.drop(['Volume', 'Open', 'Close', 'High', 'Low'], axis=1)
        print(data_dropped.head(1))
        self.states = data_dropped.values
        self.min_values = data_dropped.min(axis=0).values
        self.max_values = data_dropped.max(axis=0).values

        # Generate previous Close
        if dummy_period is not None:

            close_prices = pd.DataFrame()
            close_prices['Close'] = self.data["Close"]
            for i in range(1, dummy_period + 1):
                close_prices['Close (n - %s)' % i] = self.data['Close'].shift(i)

            self.close = close_prices.values

        self._reset()

    def _reset(self, train=True):

        if train:
            obs = self.states[0]
            self.current_index = 1
            self._end = self.train_end_index
        else:
            self.current_index = self.train_end_index + 1
            obs = self.states[self.current_index]
            self._end = self.count - 1

        self._data = self.data.iloc[self.current_index:self._end + 1]

        return obs

    def _step(self, open_trade, duration_trade):
        if open_trade:
            obs = self.states[self.current_index] + [open_trade] + [duration_trade]
        else:
            obs = self.states[self.current_index]
        self.current_index += 1
        done = self.current_index > self._end
        return obs, done
