import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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
    
    
    processed_obs = processed_obs.astype(np.float).ravel()
   
    if prev_obs is not None:
        input_obs = processed_obs - prev_obs
    else:
        input_obs = np.zeros(input_dims)
    prev_obs = processed_obs
    return input_obs, prev_obs
   
def init_weights(inp_layer , hidden_layer , output_layer):
    W1 = np.random.randn(hidden_layer , inp_layer)/np.sqrt(inp_layer)
    W2 = np.random.randn(output_layer , hidden_layer)/np.sqrt(hidden_layer)
    
    weights = {
        'W1': W1,
        'W2': W2}
        
    return weights      
    
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def relu(x):
    vector[x < 0] = 0
    return vector
    
def forward_prop(x , weights):
    Z1 = np.dot(weights['W1'] , x)
    A1 = relu(Z1)
    
    Z2 = np.dot(weights['W2'] , A1)
    A2 = sigmoid(Z2)
    
    return A2
    
def action(threshold):
    prob = np.random.uniform()
    
    if prob < threshold:
        return 2 #move up
        
    else
        return 3        
    
        

#observation = env.reset()
#processed_obs = process_obs(observation , None ,10)
#print(processed_obs.shape)
#plt.imshow(processed_obs)
#plt.show()


def main():
    env = gym.make('Pong-v0')
    observation = env.reset()
    
    batch_size = 5 # how many episodes to wait before moving the weights
    gamma = 0.99 # discount factor for reward
    decay_rate = 0.99
    hidden_layer_neurons = 200 # number of neurons
    input_dimensions = 6400 # dimension of our observation images
    learning_rate = 1e-4
    
    
    episode_number = 0
    reward_sum = 0
    running_reward = None
    prev_obs = None
    
    
    weights = init_weights(input_dimensions , hidden_layer_neurons , 1)
    
    while True:
        env.render()
        processed_obs , prev_obs = process_obs(observation , prev_obs , input_dimensions)


        up_prob = forward_prop(processed_obs , weights)
        action = action(up_prob)
        
        observation, reward, done, info = env.step(action)
        reward_sum += reward
    
    




