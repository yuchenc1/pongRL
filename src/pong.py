from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
# import cPickle as pickle
import pickle
import shutil

import gym
import numpy as np
import tensorflow as tf

from utils import *

DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
DEFINE_string("data_path", "", "")
DEFINE_string("output_dir", "", "")

DEFINE_integer("n_eps", 100, "Number of training episodes")
DEFINE_integer("n_hidden", 200, "Hidden dimension")
DEFINE_float("lr", 0.001, "Learning rate")
DEFINE_float("discount", 0.99, "Discount factor")
DEFINE_integer("batch_size", 1, "Number of states fed per batch")
DEFINE_boolean("render", False, "render game")
DEFINE_integer("save_step", 10, "steps of epoch to save checkpoint")
DEFINE_string("checkpoint_dir", "./checkpoint/policy_network.ckpt", "path of the checkpoint")
DEFINE_boolean("load_checkpoint", False, "whether to load checkpoint")

flags = tf.app.flags
FLAGS = flags.FLAGS

UP_ACTION = 2
DOWN_ACTION = 3
action_dict = {DOWN_ACTION: 0, UP_ACTION: 1}

# from github https://github.com/mrahtz/tensorflow-rl-pong/blob/master/pong.py
def discount_rewards(rewards, discount_factor):
    discounted_rewards = np.zeros_like(rewards)
    for t in range(len(rewards)):
        discounted_reward_sum = 0
        discount = 1
        for k in range(t, len(rewards)):
            discounted_reward_sum += rewards[k] * discount
            discount *= discount_factor
            if rewards[k] != 0:
                # Don't count rewards from subsequent rounds
                break
        discounted_rewards[t] = discounted_reward_sum
    return discounted_rewards

# From Andrej's code
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

def forward_pass(sess, up_probability, observations, observation_delta):
    res = sess.run(up_probability, feed_dict={observations: observation_delta.reshape([1, -1])})
    return res

def train_steps(sess, observations, sampled_actions, advantage, train_op, state_action_reward_tuples):
    states, actions, rewards = zip(*state_action_reward_tuples)
    states = np.vstack(states)
    actions = np.vstack(actions)
    rewards = np.vstack(rewards)
    feed_dict = {
        observations: states,
        sampled_actions: actions,
        advantage: rewards
    }
    sess.run(train_op, feed_dict)

def train(hparams):
    print("-" * 80)
    print("Building OpenAI gym environment.")
    env = gym.make("Pong-v0")

    sess = tf.InteractiveSession()

    observations = tf.placeholder(tf.float32, [None, 6400])

    sampled_actions = tf.placeholder(tf.float32, [None, 1])

    advantage = tf.placeholder(tf.float32, [None, 1], name='advantage')

    h = tf.layers.dense(observations, units=hparams.n_hidden, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())

    up_probability = tf.layers.dense(h, units=1, activation=tf.sigmoid, kernel_initializer=tf.contrib.layers.xavier_initializer())

    loss = tf.losses.log_loss(labels=sampled_actions, predictions=up_probability, weights=advantage)

    optimizer = tf.train.AdamOptimizer(hparams.lr)

    train_op = optimizer.minimize(loss)

    tf.global_variables_initializer().run()

    saver = tf.train.Saver()

    if hparams.load_checkpoint:
        print('I am loading!')
        saver.restore(sess, hparams.checkpoint_dir)

    batch_state_action_reward_tuples = []
    smoothed_reward = None
    episode_n = 1

    while True:
        episode_done = False
        episode_reward_sum = 0

        round_n = 1

        last_observation = env.reset()
        last_observation = prepro(last_observation)
        action = env.action_space.sample()
        observation, _, _, _ = env.step(action)
        observation = prepro(observation)
        n_steps = 1
        while not episode_done:
            if hparams.render:
                env.render()
            observation_delta = observation - last_observation
            last_observation = observation
            up_probability_current = forward_pass(sess, up_probability, observations, observation_delta)[0]

            if np.random.uniform() < up_probability_current:
                action = UP_ACTION
            else:
                action = DOWN_ACTION

            observation, reward, episode_done, info = env.step(action)
            observation = prepro(observation)
            episode_reward_sum += reward
            n_steps += 1

            tup = (observation_delta, action_dict[action], reward)

            batch_state_action_reward_tuples.append(tup)

            if reward != 0:
                round_n += 1
                n_steps = 0
        print("Episode %d finished after %d rounds" % (episode_n, round_n))

        if episode_reward_sum > 0:
            print('Win: 21    Lose: ' + str(21 - episode_reward_sum))
        else:
            print('Win: ' + str(episode_reward_sum + 21) + '    Lose: 21')

        if smoothed_reward is None:
            smoothed_reward = episode_reward_sum
        else:
            smoothed_reward = smoothed_reward * 0.99 + episode_reward_sum * 0.01

        print("Reward total was %.3f; discounted moving average of reward is %.3f" % (episode_reward_sum, smoothed_reward))

        print('-'*80)

        if episode_reward_sum > 0:
            break
        #     break

        if episode_n % hparams.batch_size == 0:
            states, actions, rewards = zip(*batch_state_action_reward_tuples)
            rewards = discount_rewards(rewards, hparams.discount)
            rewards -= np.mean(rewards)
            rewards /= np.std(rewards)
            batch_state_action_reward_tuples = list(zip(states, actions, rewards))
            train_steps(sess, observations, sampled_actions, advantage, train_op, batch_state_action_reward_tuples)
            batch_state_action_reward_tuples = []

        if episode_n % hparams.save_step == 0:
            saver.save(sess, hparams.checkpoint_dir)
            print('I am saving!')

        episode_n += 1

def main():
    # print("-" * 80)
    # if not os.path.isdir(FLAGS.output_dir):
    #     print("Path {} does not exist. Creating.".format(FLAGS.output_dir))
    #     os.makedirs(FLAGS.output_dir)
    # elif FLAGS.reset_output_dir:
    #     print("Path {} exists. Remove and remake.".format(FLAGS.output_dir))
    #     shutil.rmtree(FLAGS.output_dir)
    #     os.makedirs(FLAGS.output_dir)

    # print("-" * 80)
    # log_file = os.path.join(FLAGS.output_dir, "stdout")
    # print("Logging to {}".format(log_file))
    # sys.stdout = Logger(log_file)

    print_user_flags()

    hparams = tf.contrib.training.HParams(
        n_hidden=FLAGS.n_hidden,
        lr=FLAGS.lr,
        discount=FLAGS.discount,
        render=FLAGS.render,
        batch_size=FLAGS.batch_size,
        checkpoint_dir=FLAGS.checkpoint_dir,
        save_step=FLAGS.save_step,
        load_checkpoint=FLAGS.load_checkpoint,
    )
    train(hparams)


if __name__ == "__main__":
    main()
