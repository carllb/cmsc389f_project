import numpy as np

class Memory(object):
    def __init__(self, max_memory=100, discount=0.9):
        self.memory = [[None,None]] * max_memory
        self.size = max_memory
        self.discount = discount
        self.curr = -1
        self.full = False
    
    def remember(self, states, done):
        if (not self.full) and self.curr + 1 >= self.size:
            self.full == True
        self.curr = (self.curr + 1) % self.size
        self.memory[self.curr] = [states, done]
        
    
    def get_batch(self, model, batch_size=10):

        len_menory = self.memory
        if (not self.full):
            len_menory = self.curr + 1
        
        if (len_menory < 1):
            return None

        num_actions = model.output_shape[-1]

        env_dim = self.memory[0][0][0].shape[1]

        inputs = np.zeros((min(len_menory, batch_size), env_dim))

        targets = np.zeros((inputs.shape[0], num_actions))

        for i, idx in enumerate(np.random.randint(0, len_menory, size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            
            #We also need to know whether the game ended at this state
            done = self.memory[idx][1]

            #add the state s to the input
            inputs[i:i+1] = state_t
            
            # First we fill the target values with the predictions of the model.
            # They will not be affected by training (since the training loss for them is 0)
            targets[i] = model.predict(state_t)[0]
            
            """
            If the game ended, the expected reward Q(s,a) should be the final reward r.
            Otherwise the target value is r + gamma * max Q(s’,a’)
            """
            #  Here Q_sa is max_a'Q(s', a')
            Q_sa = np.max(model.predict(state_tp1)[0])
            
            #if the game ended, the reward is the final reward
            if done:  # if its a final state
                targets[i, action_t] = reward_t
            else:
                # r + gamma * max Q(s’,a’)
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets

    

