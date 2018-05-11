import sys
import math

import json
import matplotlib.pyplot as plt
import numpy as np
import time

# TODO: do we need these two?
from IPython import display

import seaborn

from keras.models import model_from_json
from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd

import tensorflow as tf

import gym, gym.spaces

def get_timestamp():
    return time.strftime("%d-%m-%y_%H:%M:%S")

LOSS_LOG_FILE = "loss" + get_timestamp() + ".csv"
RWD_LOG_FILE = "reward" + get_timestamp() + ".csv"
EPSILON_LOG_FILE = "epsilon" + get_timestamp() + ".csv"

seaborn.set()

env = gym.make('Breakout-ram-v0')

env.reset()


last_frame_time = 0
ada_divisor = 100
min_epsilon = 0.1

# I dont know if this works
def set_max_fps(last_frame_time,FPS = 1):
    current_milli_time = lambda: int(round(time.time() * 1000))
    sleep_time = 1./FPS - (current_milli_time() - last_frame_time)   
    if sleep_time > 0:       
        time.sleep(sleep_time)
    return current_milli_time()

def get_epsilon(t):
    return max(min_epsilon, min(1, 1.0 - math.log10((t + 1) / ada_divisor)))


class ExperienceReplay(object):
    """
    During gameplay all the experiences < s, a, r, s’ > are stored in a replay memory. 
    In training, batches of randomly drawn experiences are used to generate the input and target for training.
    """
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount
    
    def remember(self, states, done):
        self.memory.append([states, done])

        if len(self.memory) > self.max_memory:
            del self.memory[0]
    
    def get_batch(self, model, batch_size=10):

        len_memory = len(self.memory)

        num_actions = model.output_shape[-1]

        env_dim = self.memory[0][0][0].shape[1]

        inputs = np.zeros((min(len_memory, batch_size), env_dim))

        targets = np.zeros((inputs.shape[0], num_actions))

        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):

            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]

            done = self.memory[idx][1]

            inputs[i:i+1] = state_t

            targets[i] = model.predict(state_t)[0]

            Q_sa = np.max(model.predict(state_tp1)[0])

            if done:
                targets[i, action_t] = reward_t
            else:
                targets[i, action_t] = reward_t + self.discount * Q_sa

        return inputs, targets

def baseline_model(obs_size, num_action, hidden_size):

    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(obs_size,), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(sgd(lr=.000001), "mse")
    return model


#hyper parameters
#epsilon = .9
discount = 0.9
num_actions = env.action_space.n # one dimensional action space
max_memory = 500
hidden_size = 100
batch_size = 1
obs_size = len(env.observation_space.sample())

model = None
testing = False

print(sys.argv)

if len(sys.argv) > 2:
    if sys.argv[1].strip() == "load":
        model = load_model(sys.argv[2])
    elif sys.argv[1].strip() == "test":
        testing = True
        model = load_model(sys.argv[2])
else:
    model = baseline_model(obs_size, num_actions, hidden_size)

model.summary()

exp_replay = ExperienceReplay(max_memory=max_memory, discount=discount)


def test(model):

    c = 0
    input_t = np.array([env.reset()])
    points = 0
    done = False    
    while not done and c <= 1000:
        input_tm1 = input_t
        q = model.predict(input_tm1)
        action = np.argmax(q[0])

        obs, reward, done, _ = env.step(action)
        input_t = np.array([obs])
        points += reward
        env.render()
        c += 1
    return points
           


def train(model, epochs, verbose = 1, disp_every = 100):
    win_cnt = 0

    win_hist = []

    loss_log    = np.zeros(epochs)
    rwd_log     = np.zeros(epochs)
    epsilon_log = np.zeros(epochs)

    for e in range(epochs):
        loss = 0.

        input_t = np.array([env.reset()])
       
        done = False
        
        epsilon = get_epsilon(e)

        rwd = 0

        file_name_id_str = get_timestamp()

        while not done:
            input_tm1 = input_t

            # Sprinkle in some exploration

            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                q = model.predict(input_tm1)
                action = np.argmax(q[0])
            
            obs, reward, done, _ = env.step(action)

            rwd += reward

            input_t = np.array([obs])

            # don't need this
            # 20 is arbritrary
            if reward == 20:
                win_cnt += 1
            
            # Comment to speed up
            #env.render()

            exp_replay.remember([input_tm1, action, reward, input_t], done)

            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            batch_loss = model.train_on_batch(inputs, targets)

            loss += batch_loss
        
        loss_log[e] = loss
        rwd_log[e] = rwd
        epsilon_log[e] = epsilon
       
       
        if (e + 1) % disp_every == 0:
            #test(model)
            model.save("model" +  file_name_id_str  + str(e))
            np.savetxt(LOSS_LOG_FILE, loss_log, delimiter=",")
            np.savetxt(RWD_LOG_FILE, rwd_log, delimiter=",")        
            np.savetxt(EPSILON_LOG_FILE, epsilon_log, delimiter=",")

        if verbose > 0:
            print("Epoch {:03d}/{:03d} | Loss {:.4f} | Reward {} | Epsilon {}".format(e,epochs, loss, rwd, epsilon))
        win_hist.append(win_cnt)
    return win_hist

epoch = 5000



if not testing:
    hist = train(model, epoch, verbose=1)
    print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~TRANING DONE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    model.save("model_final" +  get_timestamp())

print("showing model")
best_reward = 0
while 1:
    rwd = test(model)
    if rwd > best_reward:
        print("New best reward: " + str(rwd))
        best_reward = rwd    

