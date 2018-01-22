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
explore= 5000 
observe = 500

REPLAY_MEMORY = 5000

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

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
  
def createNetwork():

    inpt = tf.placeholder("float", [None, 80, 80, 4])

    W1=  weight_variable([8, 8, 4, 32])
    b1 = tf.Variable(tf.zeros([32]))

    W2 = weight_variable([4, 4, 32, 64])
    b2 = tf.Variable(tf.zeros([64]))

    W3 = weight_variable([3, 3, 64, 64])
    b3 = tf.Variable(tf.zeros([64]))

    W4 = weight_variable([576, 784])
    b4 = tf.Variable(tf.zeros([784]))

    W5 = weight_variable([784, ACTIONS])
    b5 = tf.Variable(tf.zeros([ACTIONS]))

  
    conv1 = tf.nn.relu(tf.nn.conv2d(inpt, W1, strides = [1, 4, 4, 1], padding = "VALID") + b1)

    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W2, strides = [1, 2, 2, 1], padding = "VALID") + b2)

    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W3, strides = [1, 1, 1, 1], padding = "VALID") + b3)

    conv3_flat = tf.reshape(conv3, [-1, 576])

    fc4 = tf.nn.relu(tf.matmul(conv3_flat, W4) + b4)
    
    fc5 = tf.matmul(fc4, W5) + b5

    return inpt, fc5

def trainNetwork(inp, out, sess):

    argmax = tf.placeholder("float", [None, ACTIONS]) 
    y = tf.placeholder("float", [None]) #Target value

    action = tf.reduce_sum(tf.multiply(out, argmax), reduction_indices = 1)
    loss = tf.reduce_mean(tf.square(action - y)) #Squared error loss
    
    train_step = tf.train.AdamOptimizer(1e-6).minimize(loss)
	
    frame = env.reset()
    processed_image = process_image(frame)
    
    inp_t = np.stack((processed_image , processed_image , processed_image , processed_image) , axis=2)
    sess.run(tf.initialize_all_variables())

    t = 0
    epsilon = initial_eps
    replay_memory = []
    
    while True:
        print(inp_t.shape)
       
        out_t = out.eval(feed_dict = {inp : [inp_t]})[0]
                
        argmax_t = np.zeros([ACTIONS])
        
        if(random.random() <= epsilon):
            act= random.randrange(ACTIONS)
            
        else:
            act = np.argmax(out_t)
            
        argmax_t[act] = 1  
            
        if epsilon > final_eps:
            epsilon -= (initial_eps - final_eps) / explore  
            
        next_state , reward , done , info = env.step(act)
        
        next_image = process_image(next_state)
        next_image = np.reshape(next_image , (80,80,1))
        
        inp_t1 = np.append(next_image, inp_t[:, :, 0:3], axis = 2)  
        replay_memory.append((inp_t , argmax_t , reward , inp_t1))
        
        if len(replay_memory) > REPLAY_MEMORY:
            replay_memory.pop(0)      
        
        if t>observe:
            
            minibatch = random.sample(replay_memory , batch_size)
            
            inp_batch = [d[0] for d in minibatch]
            action_batch = [d[1] for d in minibatch]
            reward_batch = [d[2] for d in minibatch]
            next_state_batch = [d[3] for d in minibatch]
        
            y_batch = []
            out_batch = out.eval(feed_dict = {inp : next_state_batch})
            
            #add values to our batch
            for i in range(0, len(minibatch)):
                y_batch.append(reward_batch[i] + gamma * np.max(out_batch[i]))
            
            
            train_step.run(feed_dict = {
                           y : y_batch,
                           argmax : action_batch,
                           inp : inp_batch
                           })
        inp_t = inp_t1
        t = t+1  
        
        print("TIMESTEP", t, "/ EPSILON", epsilon, "/ ACTION", act, "/ REWARD", reward, "/ Q_MAX %e" % np.max(out_t))      

def main():
    sess = tf.InteractiveSession()
    #input layer and output layer by creating graph
    inp, out = createNetwork()
    print(out)
    #train our graph on input and output with session variables
    trainNetwork(inp, out, sess)

if __name__ == "__main__":
    main()

