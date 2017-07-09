import pandas as pd
import gym_trading
import gym
import sys
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=128, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


def run_test(env, act, episodes=1, final_test=False):
    obs = env._reset(train=False)
    start = env.sim.train_end_index + 1
    end = env.sim.count - 1

    for episode in range(episodes):
        done = False
        while done is False:
            action = act(obs[None])
            obs, reward, done, info = env.step(action)

        if not final_test:
            journal = pd.DataFrame(env.portfolio.journal)
            profit = journal["Profit"].sum()
            return env.portfolio.average_profit_per_trade, profit
        else:
            print("Training period  %s - %s" % (env.sim.date_time[start], env.sim.date_time[end]))
            print("Average Reward is %s" % (env.portfolio.average_profit_per_trade))

    if final_test:
        env._generate_summary_stats()


with U.make_session(8):
    csv = "/home/adrian/Escritorio/polinex/EURUSD60.csv"

    env = gym.make('trading-v0')
    env.initialise_simulator(csv, trade_period=5000, train_split=0.7)

    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=lambda name: U.BatchInput(env.observation_space.shape, name=name),
        q_func=model,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
    )

    replay_buffer = ReplayBuffer(50000)
    # Create the schedule for exploration starting from 1 (every action is random) down to
    # 0.02 (98% of actions are selected according to values predicted by the model).
    exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)
    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    episode_rewards = [0.0]
    obs = env.reset()
    l_mean_episode_reward = []
    for t in itertools.count():
        # Take action and update exploration to the newest value
        action = act(obs[None], update_eps=exploration.value(t))[0]

        new_obs, rew, done, _ = env.step(action)

        # Store transition in the replay buffer.
        replay_buffer.add(obs, action, rew, new_obs, float(done))

        obs = new_obs

        episode_rewards[-1] += rew

        is_solved = np.mean(episode_rewards[-101:-1]) > 500 or t >= 10000
        is_solved = is_solved and len(env.portfolio.journal) != 0

        if done:

            journal = pd.DataFrame(env.portfolio.journal)
            profit = journal["Profit"].sum()

            try:
                print("-------------------------------------")
                print("steps                     | {:}".format(t))
                print("episodes                  | {}".format(len(episode_rewards)))
                print("% time spent exploring    | {}".format(int(100 * exploration.value(t))))

                print("--")
                l_mean_episode_reward.append(round(np.mean(episode_rewards[-101:-1]), 1))

                print("mean episode reward       | {:}".format(l_mean_episode_reward[-1]))
                print("Total operations          | {}".format(len(env.portfolio.journal)))
                print("Avg duration trades       | {}".format(round(journal["Trade Duration"].mean(), 2)))
                print("Total profit episode      | {}".format(round(profit), 1))
                print("Avg profit per trade      | {}".format(round(env.portfolio.average_profit_per_trade, 3)))

                print("--")

                reward_test, profit = run_test(env=env, act=act)
                print("Total profit test:        > {}".format(round(profit, 2)))
                print("Avg profit per trade test > {}".format(round(reward_test, 3)))
                print("-------------------------------------")
            except Exception as e:
                print("Exception: ", e)
                # Update target network periodically.

            obs = env.reset()
            episode_rewards.append(0)



        if is_solved:
            # Show off the result
            env._generate_summary_stats()
            run_test(env, act, final_test=True)
            break

        else:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if t > 500:
                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
            if t % 500 == 0:
                update_target()


