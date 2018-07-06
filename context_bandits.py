mport tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class contextual_bandit():
    def __init__(self):
        self.state = 0
        # List out our bandits. Currently arms 4, 2, and 1 (respectively) are
        # the most optimal.
        self.bandits = np.array(
                [[ 0.2,  0.0, -0.0, -5.0 ], 
                 [ 0.1, -5.0,  1.0,  0.25], 
                 [-5.0,  5.0,  5.0,  5.0 ]])
        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]
        
    def getBandit(self):
        # Returns a random state for each episode.
        self.state = np.random.randint(0,len(self.bandits)) 
        return self.state
        
    def pullArm(self,action):
        # Get a random number.
        bandit = self.bandits[self.state,action]
        result = np.random.randn(1)
        if result > bandit:
            #return a positive reward.
            return 1
        else:
            #return a negative reward.
            return -1


class agent():
    def __init__(self, lr, s_size, a_size):
        # These lines established the feed-forward part of the network. The
        # agent takes a state and produces an action.
        self.state_in = tf.placeholder(shape=[1], dtype=tf.int32) 
        state_in_OH = tf.reshape(slim.one_hot_encoding(
            self.state_in, s_size), (1, -1))
        output = slim.fully_connected(state_in_OH, a_size, 
                biases_initializer=None, activation_fn=tf.nn.sigmoid,
                weights_initializer=tf.ones_initializer()) 
        self.output = tf.reshape(output, [-1]) 
        self.chosen_action = tf.argmax(self.output, 0)

        # The next six lines establish the training proceedure. We feed the
        # reward and chosen action into the network to compute the loss, and
        # use it to update the network.
        self.reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
        self.responsible_weight = tf.slice(self.output, self.action_holder, [1])
        self.loss = -(tf.log(self.responsible_weight)*self.reward_holder)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.update = optimizer.minimize(self.loss)


