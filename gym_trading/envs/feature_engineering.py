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

    # def _volume_engineering(self, df):
    #     # Get Volume Bases
    #     df['100d_Avg_Vol'] = df['Volume'].shift().rolling(window=100).mean() * 1.5
    #     df['100d_Max_Vol'] = df['Volume'].shift().rolling(window=100).max()
    #
    #     # # Get Spread Bases
    #     # df['Abs_Spread'] = np.abs(df['Adj Close'] - df['Open'])
    #     # df['Abs_Spread_Shift1'] = df['Abs_Spread'].shift()
    #     # df['100d_Avg_Spread'] = df['Abs_Spread_Shift1'].rolling(window=100).mean() * 1.5
    #     # df['100d_Max_Spread'] = df['100d_High'] - df['100d_Low']
    #     #
    #     # df.drop(['Abs_Spread_Shift1', 'Abs_Spread'], axis=1, inplace=True)
    #
    #     def relative_transform(num):
    #         if 0 <= num < 0.25:
    #             return 1
    #         elif 0.25 <= num < 0.5:
    #             return 2
    #         elif 0.5 <= num < 0.75:
    #             return 3
    #         elif 0.75 <= num < 1:
    #             return 4
    #         elif 1 <= num:
    #             return 5
    #         elif -0.25 <= num < 0:
    #             return -1
    #         elif -0.5 <= num < -0.25:
    #             return -2
    #         elif -0.75 <= num < -0.5:
    #             return -3
    #         elif -1 <= num < -0.75:
    #             return -4
    #         elif num < -1:
    #             return -5
    #         else:
    #             num
    #
    #     # Volume Engineering
    #     # Get volumes
    #     period_list = range(1, 5 + 1)
    #     for x in period_list:
    #         df['-' + str(x) + 'd_Vol'] = df['Volume'].shift(x)
    #
    #     # Get avg. volumes
    #     period_list = [10, 21, 63]
    #     for x in period_list:
    #         df[str(x) + 'd_Avg_Vol'] = df['Volume'].shift().rolling(window=x).mean()
    #
    #     # Get relative volumes 1
    #     period_list = range(1, 5 + 1)
    #     for x in period_list:
    #         df['-' + str(x) + 'd_Vol1'] = df['-' + str(x) + 'd_Vol'] / df['100d_Avg_Vol']
    #         df['-' + str(x) + 'd_Vol1'] = df['-' + str(x) + 'd_Vol1'].apply(relative_transform)
    #
    #     # Get relative avg. volumes 1
    #     period_list = [10, 21, 63]
    #     for x in period_list:
    #         df[str(x) + 'd_Avg_Vol1'] = df[str(x) + 'd_Avg_Vol'] / df['100d_Avg_Vol']
    #         df[str(x) + 'd_Avg_Vol1'] = df[str(x) + 'd_Avg_Vol1'].apply(relative_transform)
    #
    #     # Get relative volumes 2
    #     period_list = range(1, 5 + 1)
    #     for x in period_list:
    #         df['-' + str(x) + 'd_Vol2'] = df['-' + str(x) + 'd_Vol'] / df['100d_Max_Vol']
    #         df['-' + str(x) + 'd_Vol2'] = df['-' + str(x) + 'd_Vol2'].apply(relative_transform)
    #
    #     return df
    #

    #
    #
    #

    #
    # def _calculate_wicks(self, df):
    #
    #     def upperwick(open, adj_close, high):
    #         if high > open and high > adj_close:
    #             return True
    #         else:
    #             return False
    #
    #     def lowerwick(open, adj_close, low):
    #         if low < open and low < adj_close:
    #             return True
    #         else:
    #             return False
    #
    #     start_time = time.time()
    #
    #     period_list1 = range(1, 5 + 1)
    #     period_list2 = [10, 21, 63, 100]
    #     for x in period_list1:
    #         df.ix[:, '-' + str(x) + 'd_upperwick_bool'] = df.apply(
    #             lambda row: upperwick(row['-' + str(x) + 'd_Open'], row['-' + str(x) + 'd_Close'],
    #                                   row['-' + str(x) + 'd_High']), axis=1)
    #         df.ix[:, '-' + str(x) + 'd_lowerwick_bool'] = df.apply(
    #             lambda row: lowerwick(row['-' + str(x) + 'd_Open'], row['-' + str(x) + 'd_adjClose'],
    #                                   row['-' + str(x) + 'd_Low']), axis=1)
    #
    #     for x in period_list2:
    #         df.ix[:, str(x) + 'd_upperwick_bool'] = df.apply(
    #             lambda row: upperwick(row['-' + str(x) + 'd_Open'], row['-1d_adjClose'], row[str(x) + 'd_High']),
    #             axis=1)
    #         df.ix[:, str(x) + 'd_lowerwick_bool'] = df.apply(
    #             lambda row: lowerwick(row['-' + str(x) + 'd_Open'], row['-1d_adjClose'], row[str(x) + 'd_Low']), axis=1)
    #
    #     print("Getting wicks took {} seconds.".format(time.time() - start_time))
    #
    #     def get_upperwick_length(open, adj_close, high):
    #         return high - max(open, adj_close)
    #
    #     def get_lowerwick_length(open, adj_close, low):
    #         return min(open, adj_close) - low
    #
    #     start_time = time.time()
    #
    #     # Transform upper wicks
    #     period_list1 = range(1, 5 + 1)
    #     period_list2 = [10, 21, 63]
    #
    #     for x in period_list1:
    #         has_upperwicks = df['-' + str(x) + 'd_upperwick_bool']
    #         has_lowerwicks = df['-' + str(x) + 'd_lowerwick_bool']
    #
    #         df.loc[has_upperwicks, '-' + str(x) + 'd_upperwick'] = df.loc[has_upperwicks, :].apply(
    #             lambda row: get_upperwick_length(row['-' + str(x) + 'd_Open'], row['-' + str(x) + 'd_adjClose'],
    #                                              row['-' + str(x) + 'd_High']), axis=1)
    #         df.loc[has_lowerwicks, '-' + str(x) + 'd_lowerwick'] = df.loc[has_lowerwicks, :].apply(
    #             lambda row: get_lowerwick_length(row['-' + str(x) + 'd_Open'], row['-' + str(x) + 'd_adjClose'],
    #                                              row['-' + str(x) + 'd_Low']), axis=1)
    #
    #         # Get relative upperwick length
    #         df.loc[df['-' + str(x) + 'd_upperwick_bool'], '-' + str(x) + 'd_upperwick'] = df.loc[df[
    #                                                                                                  '-' + str(
    #                                                                                                      x) + 'd_upperwick_bool'], '-' + str(
    #             x) + 'd_upperwick'] / df.loc[df['-' + str(x) + 'd_upperwick_bool'], '100d_Avg_Spread']
    #         # Get relative lowerwick length
    #         df.loc[df['-' + str(x) + 'd_lowerwick_bool'], '-' + str(x) + 'd_lowerwick'] = df.loc[df[
    #                                                                                                  '-' + str(
    #                                                                                                      x) + 'd_lowerwick_bool'], '-' + str(
    #             x) + 'd_lowerwick'] / df.loc[df['-' + str(x) + 'd_lowerwick_bool'], '100d_Avg_Spread']
    #
    #         # Transform upperwick ratio to int
    #         df.loc[df['-' + str(x) + 'd_upperwick_bool'], '-' + str(x) + 'd_upperwick'] = df.loc[
    #             df['-' + str(x) + 'd_upperwick_bool'], '-' + str(x) + 'd_upperwick'].apply(relative_transform)
    #         # Transform lowerwick ratio to int
    #         df.loc[df['-' + str(x) + 'd_lowerwick_bool'], '-' + str(x) + 'd_lowerwick'] = df.loc[
    #             df['-' + str(x) + 'd_lowerwick_bool'], '-' + str(x) + 'd_lowerwick'].apply(relative_transform)
    #
    #         # Assign 0 to no-upperwick days
    #         df.loc[np.logical_not(df['-' + str(x) + 'd_upperwick_bool']), '-' + str(x) + 'd_upperwick'] = 0
    #         # Assign 0 to no-lowerwick days
    #         df.loc[np.logical_not(df['-' + str(x) + 'd_lowerwick_bool']), '-' + str(x) + 'd_lowerwick'] = 0
    #
    #     for x in period_list2:
    #         has_upperwicks = df[str(x) + 'd_upperwick_bool']
    #         has_lowerwicks = df[str(x) + 'd_lowerwick_bool']
    #
    #         df.loc[has_upperwicks, str(x) + 'd_upperwick'] = df.loc[has_upperwicks, :].apply(
    #             lambda row: get_upperwick_length(row['-' + str(x) + 'd_Open'], row['-1d_adjClose'],
    #                                              row[str(x) + 'd_High']), axis=1)
    #         df.loc[has_lowerwicks, str(x) + 'd_lowerwick'] = df.loc[has_lowerwicks, :].apply(
    #             lambda row: get_lowerwick_length(row['-' + str(x) + 'd_Open'], row['-1d_adjClose'],
    #                                              row[str(x) + 'd_Low']), axis=1)
    #
    #         # Get relative upperwick length
    #         df.loc[df[str(x) + 'd_upperwick_bool'], str(x) + 'd_upperwick'] = df.loc[df[str(
    #             x) + 'd_upperwick_bool'], str(x) + 'd_upperwick'] / df.loc[df[str(
    #             x) + 'd_upperwick_bool'], '100d_Avg_Spread']
    #         # Get relative lowerwick length
    #         df.loc[df[str(x) + 'd_lowerwick_bool'], str(x) + 'd_lowerwick'] = df.loc[df[str(
    #             x) + 'd_lowerwick_bool'], str(x) + 'd_lowerwick'] / df.loc[df[str(
    #             x) + 'd_lowerwick_bool'], '100d_Avg_Spread']
    #
    #         # Transform upperwick ratio to int
    #         df.loc[df[str(x) + 'd_upperwick_bool'], str(x) + 'd_upperwick'] = df.loc[
    #             df[str(x) + 'd_upperwick_bool'], str(x) + 'd_upperwick'].apply(relative_transform)
    #         # Transform lowerwick ratio to int
    #         df.loc[df[str(x) + 'd_lowerwick_bool'], str(x) + 'd_lowerwick'] = df.loc[
    #             df[str(x) + 'd_lowerwick_bool'], str(x) + 'd_lowerwick'].apply(relative_transform)
    #
    #         # Assign 0 to no-upperwick days
    #         df.loc[np.logical_not(df[str(x) + 'd_upperwick_bool']), str(x) + 'd_upperwick'] = 0
    #         # Assign 0 to no-lowerwick days
    #         df.loc[np.logical_not(df[str(x) + 'd_lowerwick_bool']), str(x) + 'd_lowerwick'] = 0
