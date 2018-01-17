import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import psutil
import tensorflow as tf

from matplotlib import pyplot as plt
from collections import deque, namedtuple

env = gym.make('Pong-v0')

valid_actions = [0,1,2,3]
learning_rate = 0.0001

class imageProcessor():
    def __init__(self):
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = self.input_state[35:195]
            self.output =  self.output[::2, ::2, :]
            self.output = tf.image.rgb_to_grayscale(self.output)
            
            self.output = tf.image.resize_images(self.output, [80, 80], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)


    def process(self, sess, state):

        return sess.run(self.output, feed_dict = { self.input_state: state })
        
        
ip = imageProcessor()

"""

Code to test if image is being processed properly

while True:
    state = env.reset()
    sess = tf.Session()
    for timestep in range(2000):
        env.render()
        action = env.action_space.sample()
        
        next_state , reward , done , _ = env.step(action)
        next_state = ip.process(sess , next_state)
        print(next_state.shape)
        plt.imshow(next_state)
        plt.show()
        
    if done:
        break    
        
  """                
class Estimator():

    def __init__(self, scope="estimator"):
        self.scope = scope
        with tf.variable_scope(scope):
            self._build_model()
            
    def _build_model(self):

        self.X = tf.placeholder(shape=[None, 80, 80, 4], dtype=tf.uint8, name="X")
        
        self.Y = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
   
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = tf.to_float(self.X) / 255.0
        batch_size = tf.shape(self.X)[0]

 
        conv1 = tf.contrib.layers.conv2d(X, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        self.predictions = tf.contrib.layers.fully_connected(fc1, len(valid_actions))

        
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        self.losses = tf.squared_difference(self.Y, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss)

        

    def predict(self, sess, s):  
               
         return (sess.run(self.predictions, { self.X: s }))

    def update(self, sess, s, a, y):
      
        feed_dict = { self.X: s, self.Y: y, self.actions: a }
        _, loss = sess.run([self.train_op, self.loss],feed_dict)
        
        return loss
        
    
        
  
       
"""

To check if CNN is working


tf.reset_default_graph()
global_step = tf.Variable(0, name="global_step", trainable=True)

e = Estimator(scope="test")
ip = imageProcessor()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # Example observation batch
    observation = env.reset()
    observation_p = ip.process(sess, observation)
    plt.imshow(observation_p)
    plt.show()
    observation = np.stack([observation_p] * 4, axis=2)
    observations = np.array([observation] * 2)
    
   
    # Test Prediction

    # Test training step
    y = np.array([10,10])
    a = np.array([1, 1])
    
    for i in range(10):      
        print(e.update(sess, observations, a, y))
 
     
"""

def make_policy(estimator , nA):
    
    def policy_fn(sess , observation , epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        #Predicgt q values using the neural network
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]

        bestAction = np.argmax(q_values)
        
        A[best_action] += 1-epsilon
         
        return A
        
    return policy_fn

def deep_q_learning(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    image_processor,
                    num_episodes,
                    replay_memory_size=50000,
                    replay_memory_init_size=5000,
                    update_target_estimator_every=10000,
                    discount_factor=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    batch_size=32
                    ):
                    
    policy = make_policy(q_estimator,len(VALID_ACTIONS))
    
    for epsiode in range(1 , num_epsiodes+1):
        state = env.reset()
        state = image_processor(sess , state)
        state = np.stack([state] * 4, axis=2)
        
        loss = None
        
        for t in timestep:
            
            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(valid_actions[action])
            next_state = state_processor.process(sess, next_state)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
                   
 
