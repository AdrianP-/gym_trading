# OpenAI Gym for Trading With DQN OpenAI Baseline
An example of Reinforcement Trading using OpenAI Baseline.

This is made for Adrian Portabales based on Mr. Peter Henry and inspired by Tito Ingargiola, hackthemarket.

There are some improvements in code:

* 4 States:
    * 0 = Holding the trade. Only recalculate reward
    * 1 = Buy
    * 2 = Sell
    * 3 = Close position
    
    The bot only operates with one open position at the same time. 

* At the end of every episode, a test is run. The size of train and test set in `train_split` variable 
* Now you can add more features (like indicators, momentums etc) with feature_engineering module. 

## Tutorial 

https://github.com/AdrianP-/gym_trading/blob/master/Baseline%20DQN%20Gym_Trading%20Tutorial.ipynb

## References 
OpenAI Baseline: https://github.com/openai/baselines

Gym_trading by Peter Henry: https://github.com/Henry-bee/gym_trading/

Also, good thougths in Trading and Q learning: https://github.com/savourylie/Stock-Price-Forecaster

Thanks to all!

## Requirements

Requirements:
-Pandas
-Matplotlib
-Numpy
-gym
-TA-Lib
-baselines

This framework was built in Python 3.5.2

## Contact
Twitter: @porta4k

Mail: adrianportabales@gmail.com
