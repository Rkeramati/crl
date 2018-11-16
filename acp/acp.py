import acp.acpBrain as Brain
import acp.acpMemory as Memory
import acp.acpVision as Vision
import numpy as np
import tensorflow as tf
import os
import pickle

class acp():
    def __init__(self, sess, config):
        with tf.variable_scope('acp'):
            self.brain = Brain.acpBrain(config)
            self.memory = Memory.acpMemory(self.brain.getInputSize(),\
                self.brain.getLabelSize(), config)
            self.vision = Vision.acpVision()
            sess.run(tf.global_variables_initializer())

        self.global_step = 0
        self.inference_number = 0

        self.config = config
        self.observation = []
        self.nObs = config.acpNStates
        self.nA = config.nActions
        self.saveFreq = config.acpSaveFreq
        self.summaryFreq = config.acpSummaryFreq
        self.sess = sess

    def setdir(self, model_dir):
        self.model_dir = model_dir
        self.savedir = './checkpoints/acp/%s'%(self.model_dir)
        self.logdir = './logs/%s'%(self.model_dir)+'/'
        self.outputdir = './outputs/%s'%(self.model_dir)
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)


        # Writer and Saver for acp:
        with tf.variable_scope('acp'):
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)

        acp_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='acp')
        self._saver = tf.train.Saver(acp_var_list, max_to_keep = self.config.max_checkpoint)

        self.summaryOp = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES, scope='acp')
        self.writer = tf.summary.FileWriter(self.logdir, self.sess.graph)

    def load(self):
        # Load the weights if exist:
        ckpt = tf.train.get_checkpoint_state(self.savedir)
        if ckpt and ckpt.model_checkpoint_path: #when it is not training
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(self.savedir, ckpt_name)
            self._saver.restore(self.sess, fname)
            print(" [*] ACP Load SUCCESS: %s" % fname)
        else:
            print(" [!] ACP load failed")

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


    def observe(self, s, a, ns, step):
        # sess: tf session, state: s, action: a, next state: ns tuple
        # preprocess both s and ns:
        s = self.vision.process(s)
        ns = self.vision.process(ns)
        self.global_step = step

        if len(self.observation) >= self.nObs:
            self.observation.pop(0)
            self.observation.append((s, a, ns))
        else:
            self.observation.append((s, a, ns))

        nnInput, nnLabel = self.makeInputLabel()
        self.memory.add(nnInput, nnLabel)
        _, _, entropy = self.brain.infer(self.sess, nnInput)
        return self.int_reward(entropy)

    def int_reward(self, entropy):
        reward = 1.0/(self.config.lambd * entropy + self.config.beta)
        return reward

    def sample_inference(self):
        self.inference_number += 1
        print('[ACP] Write a sample inference at %d step'%(self.global_step))

        nnInput, nnLabel = self.memory.sample()
        output, probability, entropy = self.brain.infer(self.sess, nnInput)
        data = {'input': nnInput, 'label':nnLabel, 'output': output,\
                'prob':probability, 'entropy':entropy}

        f = open(self.outputdir+'inference_step_%d_%d.pkl'\
                %(self.global_step, self.inference_number), 'wb')
        pickle.dump(data, f)
        f.close()

    def train(self):
        nnInput, nnLabel = self.memory.sample()
        summary, train_step, loss, accuracy = self.brain.train(self.sess,\
                nnLabel, nnInput, self.summaryOp)

        if int(self.global_step)%self.summaryFreq == 0:
            self.writer.add_summary(summary, self.global_step)
        if int(self.global_step)%self.saveFreq == 0:
            self._saver.save(self.sess, self.savedir+'/models', global_step=self.global_step)
        return loss, train_step
