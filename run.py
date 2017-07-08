import pandas as pd

import gym
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


def run_test(env, act, obs, episodes=1):
    env._reset(train=False)

    start = 0
    end = env.sim.train_end_index
    print("Training period  %s - %s" % (env.sim.date_time[start], env.sim.date_time[end]))

    for episode in range(episodes):

        start_index = env.sim.current_index

        # Initialise state
        start = env.sim.states[start_index - 1]
        actions = []
        done = False
        while done is False:
            action = act(obs[None])

            obs, reward, done, info = env.step(action)

            actions.append(action)

        start = env.sim.train_end_index + 1
        end = env.sim.count - 1
        print("End of Test Period from %s to %s, Average Reward is %s" % (
            env.sim.date_time[start], env.sim.date_time[end],
            env.portfolio.average_profit_per_trade))

    env._generate_summary_stats()


with U.make_session(8):
    # csv = "/home/adrian/Escritorio/polinex/LTCBTC_cutted2.csv"
    csv = "/home/adrian/Escritorio/polinex/EURUSD60.csv"

    env = gym.make('trading-v0')
    env.initialise_simulator(csv, ATR=True, trade_period=5, train_split=0.7)
    print(env.sim.states)
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
    for t in itertools.count():
        # Take action and update exploration to the newest value
        action = act(obs[None], update_eps=exploration.value(t))[0]

        new_obs, rew, done, _ = env.step(action)
        # Store transition in the replay buffer.
        replay_buffer.add(obs, action, rew, new_obs, float(done))
        # print("obs: ", obs, "new_obs", new_obs)

        obs = new_obs
        # print(rew)

        # print("action: ", action, "rew: ", rew, "episode_rewards: ", episode_rewards[-1])
        # print(episode_rewards)
        # print(env.portfolio.total_reward)



        if env.portfolio.journal:
            journal = pd.DataFrame(env.portfolio.journal)
            profit = journal["Profit"].sum()
            episode_rewards[-1] += profit
            is_solved = profit > 1000 or t == 1000
        else:
            episode_rewards[-1] += 0
            profit = None
            is_solved = False

        if done:
            # print("obs", str(obs))
            # print("new_obs", str(new_obs))
            # print("rew", rew)
            # print("action", action)
            print("-------------------------------------")
            print("steps                     | {:}".format(t))
            print("episodes                  | {}".format(len(episode_rewards)))
            print("% time spent exploring    | {}".format(int(100 * exploration.value(t))))
            print("--")
            print("mean episode reward       | {:}".format(round(np.mean(episode_rewards[-101:-1]), 1)))
            if profit:
                print("Profit all                | {}".format(round(profit), 1))
            print("Total operations          | {}".format(len(env.portfolio.journal)))
            print("--")
            print("Reward in episode         | {}".format(env.portfolio.average_profit_per_trade))
            print("-------------------------------------")

            obs = env.reset()

            episode_rewards.append(0)

        # print("End of Episode %s, Reward is %s" % (t + 1, env.portfolio.average_profit_per_trade))

        if is_solved:
            # Show off the result
            env._generate_summary_stats()
            run_test(env, act, obs)
            exit(0)

        else:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if t > 500:
                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
            # Update target network periodically.
            if t % 500 == 0:
                update_target()

print("\n\n\n\n\n")
