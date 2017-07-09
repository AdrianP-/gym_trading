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
from .feature_engineering import FeatureEngineering
from .simulator import Simulator
from .portfolio import Portfolio


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        pass

    def initialise_simulator(self, csv_name, trade_period=1000, train_split=0.8, dummy_period=None):
        self.sim = Simulator(csv_name,  train_split=train_split, dummy_period=dummy_period)
        self.portfolio = Portfolio(prices=self.sim.data[['Open', 'Close']], trade_period=trade_period,
                                   train_end_index=self.sim.train_end_index)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(self.sim.min_values, self.sim.max_values)

    def _step(self, action):
        # Return the observation, done, reward from Simulator and Portfolio
        reward, info = self.portfolio._step(action)
        obs, done = self.sim._step(self.portfolio.open_trade, self.portfolio.curr_trade["Trade Duration"])
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
