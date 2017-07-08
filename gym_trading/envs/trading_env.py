import itertools

import gym
import matplotlib.dates as mdates
import matplotlib.finance as mf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import talib
from gym import error, spaces, utils
from gym.utils import seeding


class Portfolio(object):
    '''
    Parameters:
    trading_cost: Cost of taking a position is 3 pips by default
    _prices: Dataframe of Open and Close prices for reward calculation
    reward_normalizer: to convert precision of prices (0.0001) to number (1) for reward calculation
    total_reward: Keep track of reward accumulated in an episode
    current_time: the current index in the dataframe
    step: keep track of number of steps taken. Different from current_time

    curr_trade: A dictionary that records the details of an open position
    journal: Stores all records of curr_trade
    holding_trade: Boolean Flag to allow new position

    '''

    def __init__(self, prices, train_end_index, trade_period=1, max_price=10, denom=0.0001, cost=3):

        self.train_end = train_end_index
        self.trade_period = trade_period

        # Trading cost is 3 pips
        self.trading_cost = cost

        # Store list of Open price and Close price to manage reward calculation
        self._open = prices.Open.values
        self._close = prices.Close.values
        self._index = prices.index

        # To normalise reward terms
        self.reward_normalizer = 1. / denom
        self._reset()
        self.open_trade = False

    def _reset(self, train=True):
        self.total_reward = 0
        self.total_trades = 0
        self.average_profit_per_trade = 0
        self.count_open_trades = 0

        if train:
            self.current_time = 1
        else:
            self.current_time = self.train_end + 1

        self.curr_trade = {'Entry Price': 0, 'Exit Price': 0, 'Entry Time': None, 'Exit Time': None, 'Profit': 0,
                           'Trade Duration': 0, 'Type': None, 'reward': 0}
        self.journal = []

        self.open_trade = False

    def _reset_trade(self):
        self.curr_trade = {'Entry Price': 0, 'Exit Price': 0, 'Entry Time': None, 'Exit Time': None, 'Profit': 0,
                           'Trade Duration': 0, 'Type': None, 'reward': 0}

    def close_trade(self, curr_close_price, curr_time):

        if self.curr_trade['Type'] == 'SELL':
            self.count_open_trades -= 1
            # Update remaining keys in curr_trade dict
            self.curr_trade['Exit Price'] = curr_close_price
            self.curr_trade['Exit Time'] = curr_time
            self.curr_trade['Profit'] = -1 * (
                curr_close_price - self.curr_trade['Entry Price']) * self.reward_normalizer - self.trading_cost

            # Add the current trade to the journal
            self.journal.append(self.curr_trade)
            self._reset_trade()
            self.open_trade = False

        if self.curr_trade['Type'] == 'BUY':
            self.count_open_trades -= 1
            # Action is 1, Selling to close the Long position

            # Update remaining  keys in curr_trade dict
            self.curr_trade['Exit Price'] = curr_close_price
            self.curr_trade['Exit Time'] = curr_time
            self.curr_trade['Profit'] = (curr_close_price - self.curr_trade[
                'Entry Price']) * self.reward_normalizer - self.trading_cost

            # Add curr_trade to journal, then reset curr_trade
            self.journal.append(self.curr_trade)
            self._reset_trade()

            self.open_trade = False

    def _holding_trade(self, curr_close_price, prev_close_price, reward):
        self.curr_trade['Trade Duration'] += 1

        if self.curr_trade['Type'] == 'SELL':
            reward = -1 * (curr_close_price - prev_close_price) * self.reward_normalizer
        if self.curr_trade['Type'] == 'BUY':
            reward = (curr_close_price - prev_close_price) * self.reward_normalizer

        return reward

    def _step(self, action):
        curr_open_price = self._open[self.current_time]
        curr_close_price = self._close[self.current_time]
        curr_time = self._index[self.current_time]
        prev_close_price = self._close[self.current_time - 1]
        reward = 0

        if action == 3 or self.curr_trade['Trade Duration'] >= self.trade_period:
            # Closing trade or trade duration is reached
            self.close_trade(curr_close_price, curr_time)

        elif action == 1:
            if not self.open_trade:
                # BUYING
                self.curr_trade['Entry Price'] = curr_open_price
                self.curr_trade['Type'] = "BUY"
                self.curr_trade['Entry Time'] = curr_time
                self.curr_trade['Trade Duration'] += 1
                reward = (curr_close_price - curr_open_price) * self.reward_normalizer - self.trading_cost
                self.total_trades += 1
                self.open_trade = True
                self.count_open_trades += 1
            else:
                reward = self._holding_trade(curr_close_price, prev_close_price, reward)

        elif action == 2:
            if not self.open_trade:
                # SELLING
                self.curr_trade['Entry Price'] = curr_open_price
                self.curr_trade['Type'] = "SELL"
                self.curr_trade['Entry Time'] = curr_time
                self.curr_trade['Trade Duration'] += 1
                reward = -1 * (curr_close_price - curr_open_price) * self.reward_normalizer - self.trading_cost
                self.total_trades += 1
                self.open_trade = True
                self.count_open_trades += 1
            else:
                reward = self._holding_trade(curr_close_price, prev_close_price, reward)

        elif action == 0:
            # Holding trade
            if self.open_trade:
                reward = self._holding_trade(curr_close_price, prev_close_price, reward)
            else:
                pass

        self.curr_trade["reward"] += reward
        self.total_reward += reward

        if self.total_trades > 0:
            self.average_profit_per_trade = self.total_reward / self.total_trades

        self.current_time += 1

        info = {'Average reward per trade': self.average_profit_per_trade,
                'Reward for this trade': self.curr_trade["reward"],
                'Total reward': self.total_reward}

        return self.curr_trade["reward"], info





class Simulator(object):
    def __init__(self, csv_name, train_split, dummy_period=None, ATR=False, train=True, multiple_trades=False):
        if "EUR" in csv_name:
            df = pd.read_csv(csv_name, parse_dates=[[0, 1]], header=None,
                             names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df = df[~np.isnan(df['Open'])].set_index('Date_Time')

        else:
            df = pd.read_csv(csv_name, usecols=['Date', 'High', 'Low', 'Open', 'Close', 'Volume'])
            df = df[~np.isnan(df['Open'])].set_index('Date')

        df = self.feature_engineering(ATR, df)

        ##Attributes
        self.data = df
        self.date_time = df.index
        self.count = df.shape[0]
        self.train_end_index = int(train_split * self.count)

        # Attributes related to the observation state: Return
        # print(self.data.head(1))

        data_dropped = self.data.drop(['Open', 'Close', 'High', 'Low'], axis=1)
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

    def feature_engineering(self, ATR, df):

        # Compute Returns based on consecutive closed prices
        df['Return'] = df['Close'].pct_change() * 100

        # Normalization of returns
        mean = df['Return'].mean()
        std = df['Return'].std()
        df['Return'] = (df['Return'] - np.array(mean)) / np.array(std)

        df = self._generate_indicators(df, ATR)
        df = self.price_engineering(df)
        df = self.volume_engineering(df)
        
        df['Open Trade'] = np.zeros(df.shape[0])

        df = df.dropna()
        return df

    def price_engineering(self, df):
        # Price Engineering
        # Get opens
        period_list = [1, 2, 3, 4, 5, 10, 21, 63]
        for x in period_list:
            df['-' + str(x) + 'd_Open'] = df['Open'].shift(x)
        #
        # # Get adjCloses
        # period_list = range(1, 5 + 1)
        # for x in period_list:
        #     df['-' + str(x) + 'd_adjClose'] = df['Adj Close'].shift(x)

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

    def volume_engineering(self, df):
        # Get Volume Bases
        df['100d_Avg_Vol'] = df['Volume'].shift().rolling(window=100).mean() * 1.5
        df['100d_Max_Vol'] = df['Volume'].shift().rolling(window=100).max()

        # # Get Spread Bases
        # df['Abs_Spread'] = np.abs(df['Adj Close'] - df['Open'])
        # df['Abs_Spread_Shift1'] = df['Abs_Spread'].shift()
        # df['100d_Avg_Spread'] = df['Abs_Spread_Shift1'].rolling(window=100).mean() * 1.5
        # df['100d_Max_Spread'] = df['100d_High'] - df['100d_Low']
        #
        # df.drop(['Abs_Spread_Shift1', 'Abs_Spread'], axis=1, inplace=True)

        def relative_transform(num):
            if 0 <= num < 0.25:
                return 1
            elif 0.25 <= num < 0.5:
                return 2
            elif 0.5 <= num < 0.75:
                return 3
            elif 0.75 <= num < 1:
                return 4
            elif 1 <= num:
                return 5
            elif -0.25 <= num < 0:
                return -1
            elif -0.5 <= num < -0.25:
                return -2
            elif -0.75 <= num < -0.5:
                return -3
            elif -1 <= num < -0.75:
                return -4
            elif num < -1:
                return -5
            else:
                num

        # Volume Engineering
        # Get volumes
        period_list = range(1, 5 + 1)
        for x in period_list:
            df['-' + str(x) + 'd_Vol'] = df['Volume'].shift(x)

        # Get avg. volumes
        period_list = [10, 21, 63]
        for x in period_list:
            df[str(x) + 'd_Avg_Vol'] = df['Volume'].shift().rolling(window=x).mean()

        # Get relative volumes 1
        period_list = range(1, 5 + 1)
        for x in period_list:
            df['-' + str(x) + 'd_Vol1'] = df['-' + str(x) + 'd_Vol'] / df['100d_Avg_Vol']
            df['-' + str(x) + 'd_Vol1'] = df['-' + str(x) + 'd_Vol1'].apply(relative_transform)

        # Get relative avg. volumes 1
        period_list = [10, 21, 63]
        for x in period_list:
            df[str(x) + 'd_Avg_Vol1'] = df[str(x) + 'd_Avg_Vol'] / df['100d_Avg_Vol']
            df[str(x) + 'd_Avg_Vol1'] = df[str(x) + 'd_Avg_Vol1'].apply(relative_transform)

        # Get relative volumes 2
        period_list = range(1, 5 + 1)
        for x in period_list:
            df['-' + str(x) + 'd_Vol2'] = df['-' + str(x) + 'd_Vol'] / df['100d_Max_Vol']
            df['-' + str(x) + 'd_Vol2'] = df['-' + str(x) + 'd_Vol2'].apply(relative_transform)


    def _generate_indicators(self, data, ATR):
        _high = data.High.values
        _low = data.Low.values
        _close = data.Close.values

        if ATR:
            # Compute the ATR and perform Normalisation
            data['ATR'] = talib.ATR(_high, _low, _close, timeperiod=14)
            data.dropna(inplace=True)
            data['ATR'] = (data['ATR'] - np.mean(data['ATR'])) / np.std(data['ATR'])

        return data

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

    def _step(self, open_trade):
        if open_trade:
            obs = self.states[self.current_index] + [open_trade]
        else:
            obs = self.states[self.current_index]
        self.current_index += 1
        done = self.current_index > self._end
        return obs, done


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        print('Please invoke .initialise_simulator() method next to complete initialization')

    def initialise_simulator(self, csv, ATR, trade_period=1, train_split=0.8, dummy_period=None):
        self.sim = Simulator(csv, ATR=ATR, train_split=train_split, dummy_period=dummy_period)
        self.portfolio = Portfolio(prices=self.sim.data[['Open', 'Close']], trade_period=trade_period,
                                   train_end_index=self.sim.train_end_index)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(self.sim.min_values, self.sim.max_values)

    def _step(self, action):
        # Return the observation, done, reward from Simulator and Portfolio
        reward, info = self.portfolio._step(action)
        obs, done = self.sim._step(self.portfolio.open_trade)
        return obs, reward, done, info

    def _reset(self, train=True):
        obs = self.sim._reset(train)
        self.portfolio._reset(train)
        return obs

    def _generate_summary_stats(self):
        print("SUMMARY STATISTICS")

        journal = pd.DataFrame(self.portfolio.journal)
        print("Total Trades Taken: ", journal.shape[0])
        print("Total Reward: ", journal['Profit'].sum())
        print("Average Reward per Trade: ", journal['Profit'].sum() / journal['Profit'].count())
        print("Win Ratio: %s %%" % (((journal.loc[journal['Profit'] > 0, 'Profit'].count()) / journal.shape[0]) * 100))

        fig, ax = plt.subplots(figsize=(40, 10))

        data = self.sim._data
        # Get a OHLC list with tuples (dates, Open, High, Low, Close)
        ohlc = list(
            zip(mdates.date2num(data.index.to_pydatetime()), data.Open.tolist(), data.High.tolist(), data.Low.tolist(),
                data.Close.tolist()))

        # Filter out buy and sell orders for plotting
        buys = journal.loc[journal.Type == 'BUY', :]
        sells = journal.loc[journal.Type == 'SELL', :]

        # Plotting functions
        mf.candlestick_ohlc(ax, ohlc, width=0.02, colorup='green', colordown='red')
        ax.plot(buys['Entry Time'], buys['Entry Price'] - 0.001, 'b^', alpha=1.0)
        ax.plot(sells['Entry Time'], sells['Entry Price'] + 0.001, 'rv', alpha=1.0)

        plt.show()

        import pprint
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(self.portfolio.journal)
        #
        #     print("hola render")
        #     start = self.sim.train_end_index + 1
        #     end = self.sim.count - 1
        #     print("End of Test Period from %s to %s, Average Reward is %s" % (
        #         self.sim.date_time[start], self.sim.date_time[end],
        #         self.portfolio.average_profit_per_trade))
        #
        #     self.test()
        #
        # def test(self, episodes=100):
        #     self._reset(train=False)
        #     self.run_episodes(episodes, False)

        # def _render(self):
