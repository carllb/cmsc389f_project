import gym, gym.spaces, time
env = gym.make('Breakout-ram-v0')

env.reset()

done = False

while not done:
    action = env.action_space.sample()
    
    observation, reward, done, info = env.step(4)
    print (observation)
    env.render()
    time.sleep(1.0/30)