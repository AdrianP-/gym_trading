
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
