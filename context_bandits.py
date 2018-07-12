import tensorflow as tf
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
        
        with tf.variable_scope("agent"):

            with tf.variable_scope("spreading"):

                # These lines established the feed-forward part of the network.
                # The agent takes a state and produces an action.
                self.state_in = tf.placeholder(shape=[1], 
                        dtype=tf.int32, name="input") 
                state_in_OH = tf.reshape(slim.one_hot_encoding(self.state_in, 
                    s_size), (1, -1), name="HO_input")
                output = slim.fully_connected(state_in_OH, a_size, 
                        biases_initializer=None, activation_fn=tf.nn.sigmoid,
                        weights_initializer=tf.ones_initializer()) 
                self.output = tf.reshape(output, [-1], "output") 
                self.chosen_action = tf.argmax(self.output, 0,
                        name="choosen_action")

            with tf.variable_scope("updating"):

                # The next six lines establish the training proceedure. We feed
                # the reward and chosen action into the network to compute the
                # loss, and use it to update the network.
                self.reward_holder = tf.placeholder(shape=[1],
                        dtype=tf.float32, name="reward")
                self.action_holder = tf.placeholder(shape=[1],
                        dtype=tf.int32, name="action")
                #In the original tutorial erroneously called responsible_weight
                self.responsible_output = tf.slice(self.output,
                        self.action_holder, [1], "select_output")
                self.loss = -(tf.log(self.responsible_output) * 
                        self.reward_holder)
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr,
                        name="optimizer")
                self.update = optimizer.minimize(self.loss, name="update")


graph = tf.Graph()
with graph.as_default():
   
    #Load the bandits.
    cBandit = contextual_bandit()    
    
    # Load the agent.
    myAgent = agent(
            lr=0.001,
            s_size=cBandit.num_bandits,
            a_size=cBandit.num_actions)   
    
    # The weights we will evaluate to look into the network.
    weights = tf.trainable_variables()[0] 

    # Set total number of episodes to train agent on.
    total_episodes = 100000 
    # Set scoreboard for bandits to 0.
    total_reward = np.zeros([cBandit.num_bandits,cBandit.num_actions]) 
    # Set the chance of taking a random action.
    e = 0.1

    # Launch the tensorflow graph
    with tf.Session() as sess:
        
        # initialize tensorflow variables
        sess.run(tf.global_variables_initializer())
        
        # write info about the graph into the directory "./tb"
        file_writer = tf.summary.FileWriter('./tb', sess.graph)


        i = 0
        while i < total_episodes:
            
            # Get a state from the environment.
            s = cBandit.getBandit() 
            
            #Choose either a random action or one from our network.
            if np.random.rand(1) < e:
                action = np.random.randint(cBandit.num_actions)
            else:
                action = sess.run(myAgent.chosen_action,
                        feed_dict={myAgent.state_in:[s]})
            
            # Get our reward for taking an action given a bandit.
            reward = cBandit.pullArm(action) 
            
            # Update the network.
            feed_dict={myAgent.reward_holder:[reward],
                    myAgent.action_holder:[action],
                    myAgent.state_in:[s]}
            _,ww = sess.run([myAgent.update,weights], 
                    feed_dict=feed_dict)
            
             #Update our running tally of scores.
            total_reward[s,action] += reward
            if i % 5000 == 0:
                print "Mean reward for each of the " + \
                        str(cBandit.num_bandits) + \
                        " bandits: " + \
                        str(np.mean(total_reward,axis=1))
            i += 1

        print ww 

    for a in range(cBandit.num_bandits):
        print "The agent thinks action " + \
                str(np.argmax(ww[a])+1) + \
                " for bandit " + str(a+1) + \
                " is the most promising...."
        
        if np.argmax(ww[a]) == np.argmin(cBandit.bandits[a]):
            print "...and it was right!"
        else:
            print "...and it was wrong!"





