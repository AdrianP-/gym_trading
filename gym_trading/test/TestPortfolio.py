import unittest
import gym
import gym_trading

class TestPortfolio(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        csv_name = "/home/adrian/Escritorio/polinex/EURUSD60.csv"
        self.env = gym.make('trading-v0')
        self.env.initialise_simulator(csv_name, ATR=True, trade_period=5, train_split=0.7)

    def portfolio(self):
        portfolio = self.env.portfolio

        action = 1
        step_1 = portfolio._step(action=action)
        step_1_curr_trade = portfolio.curr_trade
        print(portfolio.curr_trade)


        action = 1
        step_2 = portfolio._step(action=action)
        step_2_curr_trade = portfolio.curr_trade
        print(portfolio.curr_trade)

        action = 3
        step_3 = portfolio._step(action=action)
        step_3_curr_trade = portfolio.curr_trade
        print(portfolio.curr_trade)

        print(portfolio.journal)

        res = round(step_1[0], 1)
        self.assertEqual(13.199999999999999, res)
        self.assertEqual(0, step_3[0])

    def dataframe(self):
        print(self.env.sim.data)