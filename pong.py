import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

ACTIONS_COUNT = 3
STATE_FRAMES = 4
RESIZED_SCREEN_X, RESIZED_SCREEN_Y = 80, 80

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

def init_weights():
    W1 = weight_variable([8 , 8, 4 , 32])
    b1 = bias_variable([32])
   
    W2 = weight_variable([4 , 4, 32 , 64])
    b2 = bias_variable([64]) 
    
    W3 = weight_variable([3 , 3, 64 , 64])
    b3 = bias_variable([64])
    
    W4 = weight_variable([256 , ACTIONS_COUNT])
    b4 = bias_variable([ACTIONS_COUNT])
    
    input_layer = tf.placeholder("float", [None, RESIZED_SCREEN_X, RESIZED_SCREEN_Y,STATE_FRAMES])

    hidden_layer1 = tf.nn.relu(tf.nn.conv2d(input_layer, W1, strides=[1, 4, 4, 1], padding="SAME") + b1))
    max_pool1 = tf.nn.max_pool(hidden_layer1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding="SAME")
    
    hidden_layer2 = tf.nn.relu(tf.nn.conv2d(max_pool1, W2, strides=[1, 2, 2, 1], padding="SAME") + b2))
    max_pool2 = tf.nn.max_pool(hidden_layer2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding="SAME")
    
    hidden_layer3 = tf.nn.relu(tf.nn.conv2d(max_pool2, W3, strides=[1, 1, 1, 1], padding="SAME") + b3))
    max_pool3 = tf.nn.max_pool(hidden_layer3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding="SAME")
    
    hidden_layer3_flat = tf.reshape(max_pool3, [-1, 256])
    
    output_layer = tf.matmul(hidden_layer3_flat, W4) + b4
    
    return input_layer, output_layer
    

def main():
    env = gym.make('Pong-v0')
    
    for i in range(20):
        observation = env.reset()
        previous_observation = None
    
        while True:
            env.render()
            action = env.action_space.sample()
            observation , reward , done , info = env.step(action)
            input_observation , previous_observation = process_obs(observation , previous_observation , (80,80))
            plt.imshow(input_observation)
            plt.show()
            print(previous_observation.shape)
            if done:
                break
           
        
        

main()
	
