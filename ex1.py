import gym
env = gym.make('Pong-v0')

print(env.action_space)
print(env.observation_space)

action1 = env.action_space.sample()
print(action1)    
    
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        #print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action1)
        #print("Reward: "),
        #print(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
