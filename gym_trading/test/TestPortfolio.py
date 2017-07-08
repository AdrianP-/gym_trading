import unittest
import gym
import gym_trading

class TestPortfolio(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        csv2 = "/home/adrian/Escritorio/polinex/EURUSD60.csv"
        self.env = gym.make('trading-v0')
        self.env.initialise_simulator(csv2, ATR=True, trade_period=5, train_split=0.7)

    def portfolio(self):
        action = 1
        portfolio = self.env.portfolio
        portfolio._step(action=action)
        journal = portfolio.journal
        print(journal)
        print(portfolio.curr_trade)
        action = 2
        portfolio._step(action=action)
        journal = portfolio.journal
        print(journal)
        print(portfolio.curr_trade)

