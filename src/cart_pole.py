from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import cPickle as pickle
import shutil


import gym
import numpy as np
import tensorflow as tf

from src.utils import *


DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
DEFINE_string("data_path", "", "")
DEFINE_string("output_dir", "", "")

DEFINE_integer("log_every", 1, "How many episodes to log")
DEFINE_integer("reset_every", 10, "How many episodes to reset")
DEFINE_integer("update_every", 1, "How many episodes to update params")

DEFINE_integer("n_eps", 100, "Number of training episodes")
DEFINE_integer("n_hidden", 20, "Hidden dimension")
DEFINE_float("lr", 1e-3, "Learning rate")
DEFINE_float("discount", 0.99, "Discount factor")
DEFINE_float("grad_bound", 5.0, "Gradient clipping threshold")
DEFINE_float("bl_dec", 0.99, "Baseline moving average")


flags = tf.app.flags
FLAGS = flags.FLAGS


def build_tf_graph(hparams):
  print("-" * 80)
  print("Building TF graph")

  states = tf.placeholder(tf.float32, [None] + hparams.inp_shape, name="states")
  baseline = tf.placeholder(tf.float32, None, name="baseline")
  rewards = tf.placeholder(tf.float32, [None], name="rewards")
  with tf.variable_scope("policy_net"):
    w_hidden = tf.get_variable(
      "w_hidden", [hparams.inp_shape[-1], hparams.n_hidden])
    w_soft = tf.get_variable(
      "w_soft", [hparams.n_hidden, hparams.n_actions])

  batch_size = tf.shape(states)[0]
  n_steps = tf.size(rewards)
  hidden = tf.matmul(states, w_hidden)
  hidden = tf.nn.relu(hidden)
  logits = tf.matmul(hidden, w_soft)
  actions = tf.multinomial(logits, num_samples=1)
  actions = tf.to_int32(actions)
  actions = tf.reshape(actions, [-1])
  log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=actions)

  diff = tf.to_float(n_steps) - baseline
  loss = diff * log_probs
  loss = tf.reduce_sum(loss)

  tf_vars = [var for var in tf.trainable_variables()
             if var.name.startswith("policy_net")]
  optimizer = tf.train.AdamOptimizer(learning_rate=hparams.lr)
  grads = tf.gradients(loss, tf_vars)
  grads, grad_norm = tf.clip_by_global_norm(grads, hparams.grad_bound)
  global_step = tf.train.get_or_create_global_step()
  train_op = optimizer.apply_gradients(
    zip(grads, tf_vars), global_step=global_step)

  ops = {
    "states": states,
    "rewards": rewards,
    "actions": actions,
    "global_step": global_step,
    "grad_norm": grad_norm,
    "loss": loss,
    "baseline": baseline,
    "train_op": train_op,
  }
  return ops


def train(hparams):
  print("-" * 80)
  print("Building OpenAI gym environment.")
  env = gym.make("CartPole-v0")
  n_actions = env.action_space.n
  inp_shape = list(env.observation_space.shape)
  print("Game has {0} actions".format(n_actions))
  print("Input shape: {0}".format(inp_shape))

  hparams.add_hparam("inp_shape", inp_shape)
  hparams.add_hparam("n_actions", n_actions)

  g = tf.Graph()
  with g.as_default():
    ops = build_tf_graph(hparams)

    # TF session
    saver = tf.train.Saver(max_to_keep=10)
    checkpoint_saver_hook = tf.train.CheckpointSaverHook(
      FLAGS.output_dir, save_steps=1000, saver=saver)
    hooks = [checkpoint_saver_hook]
    sess = tf.train.SingularMonitoredSession(
      hooks=hooks, checkpoint_dir=FLAGS.output_dir)

    # RL
    baseline = 0
    all_states, all_actions, all_rewards = [], [], []
    for eps in range(1, 1 + hparams.n_eps):
      states, actions, rewards = [], [], []

      state = env.reset()
      state = np.expand_dims(state, axis=0).astype(np.float32)

      for step in range(200):
        if hparams.render:
          env.render()
        states.append(state)

        action = sess.run(ops["actions"], feed_dict={ops["states"]: state})[0]
        state, reward, done, info = env.step(action)
        state = np.expand_dims(state, axis=0).astype(np.float32)

        actions.append(action)
        rewards.append(reward)
        if done:
          break

      states = np.concatenate(states, axis=0)
      rewards = np.array(rewards, dtype=np.float32)
      actions = np.array(actions, dtype=np.int32)
      baseline = (hparams.bl_dec * baseline +
                  (1.0 - hparams.bl_dec) * np.size(rewards))

      if eps % hparams.update_every == 0:
        all_states.append(states)
        all_rewards.append(rewards)
        all_actions.append(actions)
        run_ops = [
          ops["global_step"],
          ops["loss"],
          ops["grad_norm"],
          ops["train_op"],
        ]
        feed_dict = {
          ops["states"]: np.concatenate(all_states, axis=0),
          ops["rewards"]: np.concatenate(all_rewards, axis=0),
          ops["actions"]: np.concatenate(all_actions, axis=0),
          ops["baseline"]: baseline,
        }
        (global_step, loss, grad_norm,
         _) = sess.run(run_ops, feed_dict=feed_dict)

      if eps % hparams.reset_every == 0:
        all_states, all_rewards, all_actions = [], [], []

      if eps % hparams.log_every == 0:
        log_string = "eps={0:<4d}".format(eps)
        log_string += " loss={0:<10.2f}".format(loss)
        log_string += " |g|={0:<10.2f}".format(grad_norm)
        log_string += " len={0:<3d}".format(len(rewards))
        log_string += " bl={0:<5.2f}".format(baseline)
        print(log_string)

    sess.close()


def main(_):
  print("-" * 80)
  if not os.path.isdir(FLAGS.output_dir):
    print("Path {} does not exist. Creating.".format(FLAGS.output_dir))
    os.makedirs(FLAGS.output_dir)
  elif FLAGS.reset_output_dir:
    print("Path {} exists. Remove and remake.".format(FLAGS.output_dir))
    shutil.rmtree(FLAGS.output_dir)
    os.makedirs(FLAGS.output_dir)

  print("-" * 80)
  log_file = os.path.join(FLAGS.output_dir, "stdout")
  print("Logging to {}".format(log_file))
  sys.stdout = Logger(log_file)

  print_user_flags()

  hparams = tf.contrib.training.HParams(
    n_eps=FLAGS.n_eps,
    n_hidden=FLAGS.n_hidden,
    lr=FLAGS.lr,
    grad_bound=FLAGS.grad_bound,
    bl_dec=FLAGS.bl_dec,
    discount=FLAGS.discount,
    log_every=FLAGS.log_every,
    reset_every=FLAGS.reset_every,
    update_every=FLAGS.update_every,
    render=False,
  )
  train(hparams)


if __name__ == "__main__":
  tf.app.run()

