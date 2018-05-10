import gym, gym.spaces, time
env = gym.make('Breakout-ram-v0')

env.reset()

done = False

while not done:
    action = env.action_space.sample()
    print (action)
    observation, reward, done, info = env.step(action)
    
    env.render()
    time.sleep(1.0/30)