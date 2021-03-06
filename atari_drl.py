import sys
import math
import threading

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
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding, Flatten
from keras.optimizers import sgd
from keras.utils import plot_model

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

import gym, gym.spaces

from gym.envs.registration import registry, register, make, spec

import memory

register(
    id='Custom-Atari-v0',
    entry_point = 'atari_env:AtariEnv',
    kwargs={'game': 'breakout', 'obs_type': 'ram', 'repeat_action_probability': 0.25 },
    max_episode_steps=10000,
    nondeterministic=False
)

def get_timestamp():
    return time.strftime("%d-%m-%y_%H:%M:%S")

seed = int(time.clock_gettime(time.CLOCK_REALTIME))
print("Seed: " + str(seed))
np.random.seed(seed)

LOSS_LOG_FILE = "loss" + get_timestamp() + ".csv"
RWD_LOG_FILE = "reward" + get_timestamp() + ".csv"
EPSILON_LOG_FILE = "epsilon" + get_timestamp() + ".csv"
EXPERIENCE_FILE = "experience" + get_timestamp()



seaborn.set()

env = gym.make('Custom-Atari-v0')

env.reset()

last_frame_time = 0
ada_divisor = 5 #number of epochs befor epsilon begains to decrease?
min_epsilon = 0.01

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


kernel_size = 10
batch_size = 1
max_features = 256
embedding_dims = 2


def baseline_model(obs_size, num_action, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, hidden_size6):

    model = Sequential()
    model.add(Dense(hidden_size1, input_shape=(obs_size,), activation='relu'))
    #model.add(Embedding(max_features, embedding_dims, input_length=obs_size))    

    #model.add(Conv1D(filters=10,kernel_size=kernel_size, activation='relu'))
    #model.add(Conv1D(filters=hidden_size1,kernel_size=kernel_size, activation='relu'))

    #model.add(GlobalMaxPooling1D())
    #model.add(Flatten())
    model.add(Dense(hidden_size2, activation='relu'))
    model.add(Dense(hidden_size1, activation='relu'))
    model.add(Dense(hidden_size1, activation='relu'))
    model.add(Dense(hidden_size1, activation='relu'))
    model.add(Dense(hidden_size1, activation='relu'))    
    model.add(Dense(hidden_size2, activation='relu'))
    
    model.add(Dense(hidden_size6, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(sgd(lr=.0000001), "mse")
    return model


#hyper parameters
#epsilon = .9
discount = 0.99
num_actions = env.action_space.n # one dimensional action space
max_memory = 100
hidden_size1 = 256
hidden_size2 = 128
hidden_size3 = 1028
hidden_size4 = 256
hidden_size5 = 128
hidden_size6 = 16


obs_size = len(env.observation_space.sample())

model = None
testing = False

num_lives = 5

exp_replay = ExperienceReplay(max_memory=max_memory, discount=discount)
#exp_replay = memory.Memory(max_memory=max_memory, discount=discount)

if len(sys.argv) > 2:
    if sys.argv[1].strip() == "load":
        model = load_model(sys.argv[2])
        #if len(sys.argv) > 3:
            #exp_replay = pickle.load(sys.argv[3])
    elif sys.argv[1].strip() == "test":
        testing = True
        model = load_model(sys.argv[2])
else:
    model = baseline_model(obs_size, num_actions, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, hidden_size6)

model.summary()

MAX_OBS_VAL = 256

def test(model):

    c = 0
    input_t = np.array([env.reset()])/MAX_OBS_VAL
    points = 0
    done = False    
    while not done and c <= 1000:
        input_tm1 = input_t
        #if np.random.rand() <= 0.1:
        #    action = env.action_space.sample()
        #else:        
        q = model.predict(input_tm1)
        print(q)       
        action = np.argmax(q[0])
        obs, reward, done, _ = env.step(action)
        input_t = np.array([obs])/MAX_OBS_VAL
        points += reward
        env.render()
        c += 1
    return points
           
render = False

def check_input():
    while 1:
        global render        
        in_put = input()
        if (in_put == "render"):
            render = True
        else:
            render = False



def train(model, epochs, verbose = 1, disp_every = 100):
    win_cnt = 0
    global render
    win_hist = []

    loss_log    = np.zeros(epochs)
    rwd_log     = np.zeros(epochs)
    epsilon_log = np.zeros(epochs)

    for e in range(epochs):
        loss = 0.

        input_t = np.array([env.reset()])/MAX_OBS_VAL
       
        done = False
        
        epsilon = get_epsilon(e)

        rwd = 0

        file_name_id_str = get_timestamp()

        last_lives = num_lives

        while not done:
            input_tm1 = input_t

            # Sprinkle in some exploration

            #print(input_tm1)

            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                q = model.predict(input_tm1)
                action = np.argmax(q[0])
            
            obs, reward, done, info = env.step(action)

            #l = info['ale.lives']
            #if (l < last_lives):
            #    reward -= (last_lives - l)
            #    last_lives = l
            #else:
            #    reward += 1
            # cost of taking an action
            reward -= 0.01


            rwd += reward

            input_t = np.array([obs])/MAX_OBS_VAL

            # don't need this
            # 20 is arbritrary
            if reward == 20:
                win_cnt += 1
            
            # Comment to speed up
            if render:                
                env.render()
            
            
            exp_replay.remember([input_tm1, action, reward, input_t], done)

            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            input_nn = input_tm1
            target =  np.array([reward] * 4) + discount * model.predict(input_t)

            #print("NN INPUT: " + str(inputs))
            #print("NN target: " + str(targets))
            #batch_loss = model.train_on_batch(inputs, targets)
            batch_loss = model.train_on_batch(input_nn,target)

            loss += batch_loss
        
        loss_log[e] = loss
        rwd_log[e] = rwd
        epsilon_log[e] = epsilon
       
       
        if (e + 1) % disp_every == 0:
            #test(model)
            model.save("model" +  file_name_id_str  + "~" +str(e + 1))
            np.savetxt(LOSS_LOG_FILE, loss_log, delimiter=",")
            np.savetxt(RWD_LOG_FILE, rwd_log, delimiter=",")        
            np.savetxt(EPSILON_LOG_FILE, epsilon_log, delimiter=",")
            #pickle.dump(exp_replay, EXPERIENCE_FILE)
            plot_model(model, to_file='model.png')

        if verbose > 0:
            print("Epoch {:03d}/{:03d} | Loss {:.4f} | Reward {} | Epsilon {}".format(e + 1, epochs, loss, rwd, epsilon))
           
        win_hist.append(win_cnt)
    return win_hist

epoch = 5000 * 8


if not testing:
    t = threading.Thread(target=check_input)
    t.start()  
    hist = train(model, epoch, verbose=1)
    print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~TRANING DONE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    model.save("model_final" +  get_timestamp())

print("showing model")


best_reward = 0
num_runs = 0
total_reward = 0
while 1:
    num_runs += 1
    rwd = test(model)
    total_reward += rwd
    if num_runs % 100:
        print("Avrage reward: " + str(total_reward/num_runs))
    if rwd > best_reward:
        print("New best reward: " + str(rwd))
        best_reward = rwd    

