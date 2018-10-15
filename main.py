import numpy as np
import gym
import tensorflow as tf
import random
import config as Config
from acp import acp

flags = tf.app.flags

# Env:
flags.DEFINE_string('env_name', 'MontezumaRevenge-v4', 'the name of the gym envinronment')
flags.DEFINE_integer('action_repeat', 4, 'the number of action to be repeated')

flags.DEFINE_boolean('use_gpu', False, 'Whether to use gpu or not')
flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')

FLAGS = flags.FLAGS

tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

if FLAGS.gpu_fraction == '':
  raise ValueError("--gpu_fraction should be defined")

def calc_gpu_fraction(fraction_string):
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)

  fraction = 1 / (num - idx + 1)
  print(" [*] GPU : %.4f" % fraction)
  return fraction

def main(_):
    gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction))

    env = gym.make(FLAGS.env_name)
    config = Config.Config(env)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        if not tf.test.is_gpu_available() and FLAGS.use_gpu:
            raise Exception("use_gpu flag is true when no GPUs are available")

        sess.run(tf.initializers.global_variables())
        acpAgent = acp.acp(sess, config)

        step = 0
        ep = 0
        loss = []
        lrHist = []
        while ep < 20:
            terminal = False
            s = env.reset()
            while not terminal:
                action = np.random.randint(config.nActions)
                ns, r, terminal, _ = env.step(action)
                acpAgent.observe(s, action, ns)
                if ep >= 2:
                    tempLoss, step = acpAgent.train()
                    loss.append(tempLoss)
                s = ns
            ep += 1
            #summary = sess.run([summary_op])
            #train_writer.add_summary(summary, total_step)
            print('epsiode = {},  step = {}, avg loss = {}'.format(\
                    ep,  step, np.mean(loss)))

if __name__ == '__main__':
  tf.app.run()
