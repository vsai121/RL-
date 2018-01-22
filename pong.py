import gym
import itertools
import numpy as np
import os
import random
import sys
import psutil
import tensorflow as tf

env = gym.make('Pong-v0')

ACTIONS = 3 #up,down, stay

gamma = 0.99

#for updating our gradient or training over time
initial_eps= 1.0
final_eps = 0.05

#how many frames to anneal epsilon
explore= 50000 
observe = 5000

replay_memory = 50000

batch_size = 64

def downsample(image):
    return image[::2, ::2, :]

def remove_color(image):
    return image[:, :, 0]

def remove_background(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image
    
def process_image(image):
    processed_image= image[35:195] # crop
    processed_image = downsample(processed_image)
    processed_image = remove_color(processed_image)
    processed_image = remove_background(processed_image)
    
    return processed_image    

def createNetwork():

    inp = tf.placeholder("float", [None, 80, 80, 4])


    W1= tf.Variable(tf.zeros([8, 8, 4, 32]))
    b1 = tf.Variable(tf.zeros([32]))

    W2 = tf.Variable(tf.zeros([4, 4, 32, 64]))
    b2 = tf.Variable(tf.zeros([64]))

    W3 = tf.Variable(tf.zeros([3, 3, 64, 64]))
    b3 = tf.Variable(tf.zeros([64]))

    W4 = tf.Variable(tf.zeros([576, 784]))
    b4 = tf.Variable(tf.zeros([784]))

    W5 = tf.Variable(tf.zeros([784, ACTIONS]))
    b5 = tf.Variable(tf.zeros([ACTIONS]))

  
    conv1 = tf.nn.relu(tf.nn.conv2d(inp, W1, strides = [1, 4, 4, 1], padding = "VALID") + b1)

    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W2, strides = [1, 2, 2, 1], padding = "VALID") + b2)

    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W3, strides = [1, 1, 1, 1], padding = "VALID") + b3)

    conv3_flat = tf.reshape(conv3, [-1, 576])

    fc4 = tf.nn.relu(tf.matmul(conv3_flat, W4) + b4)
    
    fc5 = tf.matmul(fc4, W5) + b5

    return inp, fc5

def trainNetwork(inp, out, sess):

    argmax = tf.placeholder("float", [None, ACTIONS]) 
    y = tf.placeholder("float", [None]) #ground truth

    action = tf.reduce_sum(tf.matmul(out, argmax), reduction_indices = 1)
    loss = tf.reduce_mean(tf.square(action - y)) #Squared error loss
    
    train_step = tf.train.AdamOptimizer(1e-6).minimize(loss)
	


    frame = env.reset()
    processed_image = process_image(frame)
    
    inp_t = np.stack((processed_image , processed_image , processed_image , processed_image) , axis=2)
    
    sess.run(tf.initialize_all_variables())

    t = 0
    epsilon = initial_eps
    
    while True:
        out_t = out.eval(feed_dict = {inp : [inp_t]})[0]
    
        print(out_t.shape) 


def main():
    #create session
    sess = tf.InteractiveSession()
    #input layer and output layer by creating graph
    inp, out = createNetwork()
    #train our graph on input and output with session variables
    trainNetwork(inp, out, sess)

if __name__ == "__main__":
    main()

