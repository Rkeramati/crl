ACPHistory = 8
ACPBatch = 32
ACPLambd = 100
ACPBeta = 0.01
SAVEFREQ = 100000
SUMMARYFREQ = 200

class ACPConfig():
     def __init__(self, env):
         # General config
         self.resizeImageSize = 84
         self.nActions = env.action_size
         self.logdir = './logs'
         self.savedir = './checkpoints'
         # ACP
         self.acpBatchSize = ACPBatch
         self.acpMemorySize = int(1e4)
         self.acpNStates = ACPHistory # stacked number of frames
         self.acpLrStart = 1e-3 # initial learning rate
         self.acpLrDecayStep = 1e5 # Final learning rate
         self.acpLrDecayRate = 0.99 # schedule
         self.max_checkpoint = 30
         self.acpSaveFreq = SAVEFREQ
         self.acpSummaryFreq = SUMMARYFREQ
         self.lambd = ACPLambd
         self.beta = ACPBeta


class AgentConfig(object):
  scale = 10000
  display = False

  acpHistory = ACPHistory
  acpBatch = ACPBatch

  max_step = 100 * scale
  memory_size = 1 * scale

  batch_size = 32
  random_start = 30
  cnn_format = 'NCHW'
  discount = 0.99
  target_q_update_step = 1 * scale
  learning_rate = 0.00025
  learning_rate_minimum = 0.00025
  learning_rate_decay = 0.96
  learning_rate_decay_step = 5 * scale

  ep_end = 0.1
  ep_start = 1.
  ep_end_t = memory_size

  history_length = 4
  train_frequency = 4
  learn_start = 500 #5. * scale

  min_delta = -1
  max_delta = 1

  double_q = False
  dueling = False

  _test_step = 5 * scale
  _save_step = SAVEFREQ
  summary_freq = SUMMARYFREQ

class EnvironmentConfig(object):
  env_name = 'Breakout-v4'

  screen_width  = 84
  screen_height = 84
  max_reward = 1.
  min_reward = -1.

class DQNConfig(AgentConfig, EnvironmentConfig):
  model = ''
  pass

class M1(DQNConfig):
  backend = 'tf'
  env_type = 'detail'
  action_repeat = 1

def get_config(FLAGS):
  if FLAGS.model == 'm1':
    config = M1
  elif FLAGS.model == 'm2':
    config = M2

  for k in FLAGS.__dict__['__wrapped']:
      if k == 'use_gpu':
          if not FLAGS.__getattr__(k):
              config.cnn_format = 'NHWC'
          else:
              config.cnn_format = 'NCHW'
      if hasattr(config, k):
          setattr(config, k, FLAGS.__getattr__(k))
  return config
