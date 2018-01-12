import gym
import numpy as np
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

ACTIONS_COUNT = 3  # number of valid actions. In this case up, still and down
FUTURE_REWARD_DISCOUNT = 0.99  # decay rate of past observations
OBSERVATION_STEPS = 50000.  # time steps to observe before training
EXPLORE_STEPS = 500000.  # frames over which to anneal epsilon
INITIAL_RANDOM_ACTION_PROB = 1.0  # starting chance of an action being random
FINAL_RANDOM_ACTION_PROB = 0.05  # final chance of an action being random
MEMORY_SIZE = 500000  # number of observations to remember
MINI_BATCH_SIZE = 100  # size of mini batches
STATE_FRAMES = 4  # number of frames to store in the state
RESIZED_SCREEN_X, RESIZED_SCREEN_Y = (80, 80)
OBS_LAST_STATE_INDEX, OBS_ACTION_INDEX, OBS_REWARD_INDEX, OBS_CURRENT_STATE_INDEX, OBS_TERMINAL_INDEX = range(5)
SAVE_EVERY_X_STEPS = 10000
LEARN_RATE = 1e-6
STORE_SCORES_LEN = 200.

def downsample(image):
    return image[::2, ::2, :]

def remove_color(image):
    return image[:, :, 0]

def remove_background(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image

def process_obs(obs , prev_obs , input_dims):
    processed_obs = obs[35:195]
    processed_obs = downsample(processed_obs)
    processed_obs = remove_color(processed_obs)
    processed_obs = remove_background(processed_obs)
    processed_obs[processed_obs!= 0] = 1 


    if prev_obs is not None:
        input_obs = processed_obs - prev_obs
    else:
        input_obs = np.zeros(input_dims)
    prev_obs = processed_obs
    return input_obs, prev_obs
   
#observation = env.reset()
#processed_obs = process_obs(observation , None ,10)
#print(processed_obs.shape)
#plt.imshow(processed_obs)
#plt.show()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def init_network():
    W1 = weight_variable([8 , 8, 1 , 32])
    b1 = bias_variable([32])
   
    W2 = weight_variable([4 , 4, 32 , 64])
    b2 = bias_variable([64]) 
    
    W3 = weight_variable([3 , 3, 64 , 64])
    b3 = bias_variable([64])
    
    W4 = weight_variable([256 , ACTIONS_COUNT])
    b4 = bias_variable([ACTIONS_COUNT])
    
    input_layer = tf.placeholder("float", [None, RESIZED_SCREEN_X, RESIZED_SCREEN_Y,1])

    hidden_layer1 = tf.nn.relu(tf.nn.conv2d(input_layer, W1, strides=[1, 4, 4, 1], padding="SAME") + b1)
    max_pool1 = tf.nn.max_pool(hidden_layer1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding="SAME")
    
    hidden_layer2 = tf.nn.relu(tf.nn.conv2d(max_pool1, W2, strides=[1, 2, 2, 1], padding="SAME") + b2)
    max_pool2 = tf.nn.max_pool(hidden_layer2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding="SAME")
    
    hidden_layer3 = tf.nn.relu(tf.nn.conv2d(max_pool2, W3, strides=[1, 1, 1, 1], padding="SAME") + b3)
    max_pool3 = tf.nn.max_pool(hidden_layer3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding="SAME")
    
    hidden_layer3_flat = tf.reshape(max_pool3, [-1, 256])
    
    output_layer = tf.matmul(hidden_layer3_flat, W4) + b4
 
    return input_layer, output_layer
            
def loss(output_layer):
    targetQ = tf.placeholder(shape=[None , 3] , dtype=tf.float32) 
    loss = tf.reduce_sum(tf.square(nextQ - Qout))
    
    return loss
    
def train():    
    init = tf.global_variables_initializer()
    gamma = 0.99
    num_episodes = 2000
    
    jList = []
    rList = []   
    
    with tf.Session() as sess:
        
def main():
    env = gym.make('Pong-v0')
    input_layer , output_layer = init_network()
    
    """print(input_layer)
    print(output_layer)
    
    for _ in range(100): 
        observation = env.reset()
        previous_observation = None    
        for _  in range(100):
            env.render()
            action = env.action_space.sample()
            observation , reward , done , info = env.step(action)
            input_observation , previous_observation = process_obs(observation , previous_observation  ,(80,80))
            input_observation = input_observation[:,:,np.newaxis]""
            
            
            
main()

