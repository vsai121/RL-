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

valid_actions = [1,2,3]
learning_rate = 0.001

class imageProcessor():
    def __init__(self):
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
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
    def __init__(self , scope="estimator"):
        self.scope = scope
        
        with tf.variable_scope(scope):
            self.build_model()
            


    def build_model(self):
        self.X = tf.placeholder(shape=[None , 84 , 84 , 4] , dtype=tf.float32)
        
        #Target value
        self.Y = tf.placeholder(shape=[None] , dtype=tf.float32)
        
        self.action = tf.placeholder(shape=[None] , dtype=tf.int32)
        
        batch_size = tf.shape(self.X)[0]
        
        conv1 = tf.contrib.layers.conv2d(self.X, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu)  
         
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        self.predictions= tf.contrib.layers.fully_connected(fc1, len(valid_actions))  
   
         
        self.losses = tf.squared_difference(self.Y, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss)
               
         
    def predict(self , sess , s):
        return sess.run(self.predictions, feed_dict = {self.X: s})
        
    def update(self, sess, s, a, y):
    
        _,loss = sess.run([self.train_op, self.loss],feed_dict={self.X: s, self.Y: y, self.action:a})
        return loss 
        
        
tf.reset_default_graph()
global_step = tf.Variable(0, name="global_step", trainable=False)

e = Estimator(scope="test")
ip = imageProcessor()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # Example observation batch
    observation = env.reset()
    
    observation_p = ip.process(sess, observation)
    observation = np.stack([observation_p] * 4, axis=2)
    observations = np.array([observation] * 2)
    
    # Test Prediction
    print(e.predict(sess, observations))

    # Test training step
    y = np.array([10.0, 10.0])
    a = np.array([1, 3])
    print(e.update(sess, observations, a, y))
        
        
         
                       
