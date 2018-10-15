import acp.acpBrain as Brain
import acp.acpMemory as Memory
import acp.acpVision as Vision
import numpy as np
import tensorflow as tf
import os

class acp():
    def __init__(self, sess, config):
        with tf.variable_scope('acp'):
            self.brain = Brain.acpBrain(config)
            self.memory = Memory.acpMemory(self.brain.getInputSize(),\
                self.brain.getLabelSize(), config)
            self.vision = Vision.acpVision()

        self.observation = []
        self.nObs = config.acpNStates
        self.nA = config.nActions
        self.saveFreq = config.acpSaveFreq
        self.sess = sess
        self.savedir = config.savedir + '/acp-models'

        # Writer and Saver for acp:
        with tf.variable_scope('acp'):
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)

        acp_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='acp')
        self._saver = tf.train.Saver(acp_var_list, max_to_keep = config.max_checkpoint)

        self.summaryOp = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES, scope='acp')
        self.writer = tf.summary.FileWriter(config.logdir, self.sess.graph)


        # Load the weights if exist:
        ckpt = tf.train.get_checkpoint_state(self.savedir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(self.savedir, ckpt_name)
            self._saver.restore(self.sess, fname)
            print(" [*] ACP Load SUCCESS: %s" % fname)

    def makeInputLabel(self):
        # makes the observation into input output shape

        # last action, one-hot vector
        _, lastAction, _ = self.observation[-1]
        nnLabel = np.zeros(self.nA)
        nnLabel[lastAction] = 1

        # inialize the size
        nnInputSize = self.brain.getInputSize()
        nnInputSize.pop(0) # First is None!
        nnInput =np.zeros(tuple(nnInputSize))

        # now add all ns in the tuples in observation
        for idx in reversed(range(len(self.observation))):
            _, _, s = self.observation[idx]
            # cahnnel is the third dimension
            nnInput[:, :, idx] = s

        return nnInput, nnLabel


    def observe(self, s, a, ns):
        # sess: tf session, state: s, action: a, next state: ns tuple
        # preprocess both s and ns:
        s = self.vision.process(s)
        ns = self.vision.process(ns)

        if len(self.observation) >= self.nObs:
            self.observation.pop(0)
            self.observation.append((s, a, ns))
        else:
            self.observation.append((s, a, ns))

        nnInput, nnLabel = self.makeInputLabel()
        self.memory.add(nnInput, nnLabel)
        return self.brain.infer(self.sess, nnInput)

    def train(self):
        nnInput, nnLabel = self.memory.sample()
        summary, train_step, loss = self.brain.train(self.sess, nnLabel, nnInput, self.summaryOp)
        self.writer.add_summary(summary, train_step)
        if int(train_step)%self.saveFreq == 0:
            self._saver.save(self.sess, self.savedir+'/models', global_step=train_step)
        return loss, train_step
