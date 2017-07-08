
# coding: utf-8

# In[1]:

import gym


# # Trading Framework
# 
# This framework is developed based on Tito Ingargiola's amazing work on https://github.com/hackthemarket/gym-trading. 

# In[2]:

import gym_trading
from gym_trading.envs.Q_learning import Q


# 
# First, define the address for the CSV data
# 

# In[3]:

# csv = "/home/adrian/Escritorio/polinex/LTCBTC.csv"
csv2 = "/home/adrian/Escritorio/polinex/EURUSD60.csv"


# # Create a new OpenAI Gym environment with the customised Trading environment
# 
# 
# 
#  .initialise_simulator() must be invoked after **env.make('trading-v0')** . Within this function, provide these arguments:
# 
# 
# * **csv**: Address of the data
# 
# 
# * **ATR**: True/ False, (The only indicator available now)
# 
# 
# 
# * **trade_period**: (1 - 10), Holding period for each trades. *Default: 1*
# 
# 
# * **train_split**: (0.0,1.0), Percentage of data set for training. *Default: 0.7*

# In[4]:

env = gym.make('trading-v0')
env.initialise_simulator(csv2, ATR=True, trade_period=5, train_split=0.7)


# # States map
# 
# states_map is a discretized observation space bounded by the extreme values Return and ATR, with an interval of 0.5. For every new observation (Return, ATR) tuple pair, it is approximated to the closest pair on states_map. States_map corresponds to the row index of lookup_table

# In[5]:

print(env.sim.states)


# # Next, Create Q_learning framework
# 
# This framework wraps around the trading environment.
# 
# Arguments:
# 
# * **env**: gym_trading Environment
# 
# * **train_episodes**: Number of train episodes to update Q_table
# 
# * **learning_rate**: *Default: 0.2*
# 
# * **gamma**: *Default: 0.9* 
# 
#     Upon initializing, Q_learning has zeroed Q_table **lookup_table** and **states_map**
#     
#  
# 

# In[6]:

Q_learning = Q(env, train_episodes=100, learning_rate=0.2, gamma=0.9)


# # States_map
# **states_map** is a discretized observation space bounded by the extreme values *Return* and *ATR*, with an interval of 0.5.
# For every new observation *(Return, ATR) tuple pair*, it is approximated to the closest pair on **states_map**.  **States_map** corresponds to the row index of **lookup_table**
# 

# In[7]:

# print(Q_learning.states_map)


# # Q Table
# **lookup_table** has row size the length of **states_map** and column size of 3 (actions (0,1,2).).

# In[8]:

Q_learning.lookup_table[Q_learning.lookup_table!=0]


# All zero now, not trained yet
# 
# # Training
# 
# Filling up the Q Table

# In[9]:

Q_learning.train()


# After Training, Q Table is complete

# # Testing
# 
# Testing the new Q Table on unseen data. 
# * Q Table is not updated on Testing mode

# In[10]:

Q_learning.test(100)


# In[11]:

Q_learning._generate_summary_stats()


# All trade entries are kept in env.portfolio.journal

# In[12]:

print(env.portfolio.journal)

