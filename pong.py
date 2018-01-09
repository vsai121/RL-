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
    
    processed_obs= processed_obs.astype(np.float).ravel()
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


def main():
    env = gym.make('Pong-v0')
    
    for i in range(20):
        observation = env.reset()
        previous_observation = None
    
        while True:
            env.render()
            action = env.action_space.sample()
            observation , reward , done , info = env.step(action)
            input_observation , previous_observation = process_obs(observation , previous_observation , 6400)
            print(input_observation.shape)
            if done:
                break
           
        
        

main()
	
