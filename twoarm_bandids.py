import tensorflow as tf
import numpy as np


# List out our bandits. Currently bandit 4 (index#3) is set to most often
# provide a positive reward.
bandits = [0.2,0,-0.2,-5]
num_bandits = len(bandits)
def pullBandit(bandit):
    #Get a random number.
    result = np.random.randn(1)
    if result > bandit:
        #return a positive reward.
        return 1
    else:
        #return a negative reward.
        return -1

# Create the graph
graph = tf.Graph()
# Use it as default within a scope
with graph.as_default():
    

    # These two lines established the feed-forward part of the network. This
    # does the actual choosing.
    weights = tf.Variable(tf.ones([num_bandits]), name="weights")
    with tf.variable_scope("spreading"):
        chosen_action = tf.argmax(weights, 0, name="choosen_action")

    # The next six lines establish the training proceedure. We feed the reward
    # and chosen action into the network to compute the loss, and use it to
    # update the network.
    reward_holder = tf.placeholder(shape=[1],dtype=tf.float32, name="reward")
    action_holder = tf.placeholder(shape=[1],dtype=tf.int32, name="selected_action")
    with tf.variable_scope("updating"):
        responsible_weight = tf.slice(weights,action_holder,[1], name="selected_weight")
        loss = -(tf.log(responsible_weight)*reward_holder)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001, name="GD")
        update = optimizer.minimize(loss, name="update")
        
    total_episodes = 10000 #Set total number of episodes to train agent on.
    total_reward = np.zeros(num_bandits) #Set scoreboard for bandits to 0.
    e = 0.1 #Set the chance of taking a random action.
    
    weights_store = np.zeros([total_episodes, num_bandits])

    with tf.Session() as sess:
            
        # initialize tensorflow variables
        sess.run(tf.global_variables_initializer())
        
        # write info about the graph into the directory "./tb"
        file_writer = tf.summary.FileWriter('./tb', sess.graph)

        i = 0
        while i < total_episodes:
            
            # Choose either a random action or one from our network.
            if np.random.rand(1) < e:
                action = np.random.randint(num_bandits)
            else:
                action = sess.run(chosen_action)
            
            # Get our reward from picking one of the bandits.
            reward = pullBandit(bandits[action]) 
            
            # Update the network.
            _,resp,ww = sess.run([update,responsible_weight,weights], 
                    feed_dict={reward_holder:[reward],action_holder:[action]})
            
            # Update our running tally of scores.
            total_reward[action] += reward

            # store current weights
            weights_store[i, :] = weights.eval()

            if i % 1000 == 0:
                print "Running reward for the " + str(num_bandits) + \
                        " bandits: " + str(total_reward)
                print "Weights: {}".format(weights.eval()) 
                print
            
            i+=1

    print "The agent thinks bandit " + str(np.argmax(ww)+1) + \
            " is the most promising...."
    if np.argmax(ww) == np.argmax(-np.array(bandits)):
        print "...and it was right!"
    else:
        print "...and it was wrong!"
    

