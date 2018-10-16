import numpy as np
import tensorflow as tf

class acpBrain():
    # ACtion Predictor class
    # Neural Net for inverse Dynamic prediction
    def __init__(self, config):

        self.imageSize = config.resizeImageSize
        self.nStates = config.acpNStates
        self.nActions = config.nActions

        self._build_placeholders()
        self._build_model()
        self._build_inferences()
        self._build_optimizer(lrStart = config.acpLrStart, lrDecayRate = config.acpLrDecayRate,\
                    lrDecayStep = config.acpLrDecayStep)
    def getInputSize(self):
        return (self.input.get_shape().as_list())

    def getLabelSize(self):
        return (self.label.get_shape().as_list())

    def _build_placeholders(self):
            # input: state or number of states, s, ns : size Bx84x84xnS
            self.input = tf.placeholder(shape=[None, self.imageSize, self.imageSize, self.nStates],\
                    dtype=tf.float32, name="acp_input")
            # label = taken action, size: BxnA -- one hot label
            self.label = tf.placeholder(shape=[None, self.nActions],\
                    dtype = tf.int32, name = "acp_label")

    def _build_model(self):
            # Model architecture:
            conv1 = tf.layers.conv2d(self.input, filters=16, kernel_size=8, strides=4,\
                    padding='valid', activation=tf.nn.relu,\
                    kernel_initializer=tf.glorot_uniform_initializer(), name="conv1")

            conv2 =  tf.layers.conv2d(conv1, filters=32, kernel_size=4, strides=2,\
                    padding='valid', activation=tf.nn.relu,\
                    kernel_initializer=tf.glorot_uniform_initializer(), name="conv2")

            conv3 = tf.layers.conv2d(conv2, filters=64, kernel_size=3, strides=1,\
                    padding='valid', activation=tf.nn.relu,\
                    kernel_initializer=tf.glorot_uniform_initializer(), name="conv3")


            flatten = tf.layers.flatten(conv3, name="flat")
            dense1 = tf.layers.dense(flatten, 512, activation=tf.nn.relu,\
                    kernel_initializer=tf.glorot_uniform_initializer(), name="dense1")


            self.output = tf.layers.dense(dense1, self.nActions, activation=None, name="output")
            tf.summary.histogram('acp_output', self.output)

    def _build_inferences(self):
            # Probabilities and Entropies
            self.prob = tf.nn.softmax(self.output, axis = -1, name="softmax")
            # Compute the entropy of the probabilities
            self.H = self.prob * tf.log(self.prob)

    def _build_optimizer(self, lrStart, lrDecayStep, lrDecayRate):
        # loss function: Cross entropy with logits, sparse : exclusive classes
            crossEntropyLoss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label,\
                    logits=self.output, name="corss_entropy")
            self.loss = tf.reduce_mean(crossEntropyLoss, name="loss")
            tf.summary.scalar('acp_loss', self.loss)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.label, 1),\
                    tf.argmax(self.output, 1)), tf.float32))
            tf.summary.scalar('acp_accuracy', self.accuracy)

            # Schedueling learning rate:
            self.acp_train_step = tf.get_variable('acp_train_step',shape = [], dtype=tf.int32,\
                    initializer=tf.zeros_initializer(), trainable=False)
            tf.summary.scalar('acp_train_step', self.acp_train_step)

            self.lr = tf.train.exponential_decay(lrStart, self.acp_train_step,\
                    lrDecayStep, lrDecayRate, staircase=True)
            tf.summary.scalar('acp_lr', self.lr)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name="Adam")
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.acp_train_step)

    def predict(self, sess, nnInput):
        # Query the network for a given input x
        # Output: action index
        output = sess.run(self.output, feed_dcit={self.input : nnInput})
        return np.argmax(output, axis = 1)

    def infer(self, sess, nnInput):
        # Query Inferences and prediction
        # Output: prediction, probabilities, entropy

        if nnInput.ndim == 3:
            nnInput = np.expand_dims(nnInput, axis = 0) #batch size 1

        output, probability, entropy = sess.run([self.output, self.prob, self.H],\
                feed_dict={self.input : nnInput})
        return output, probability, entropy

    def train(self, sess, label, nnInput, summaryOp):
        # Train neural net on a batch of n=inputs
        # Outputs: loss, learning rate, step
        feed_dict = {self.input : nnInput, self.label: label}
        summary, step, loss, _, lr, accuracy = sess.run([summaryOp, self.acp_train_step,\
                self.loss, self.train_op, self.lr, self.accuracy], feed_dict=feed_dict)
        return summary, step, loss, accuracy
