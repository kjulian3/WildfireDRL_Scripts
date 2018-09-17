import numpy as np

DEFAULT_CAPACITY         = 70000   # Max size of replay memory
INIT_SIZE                = 35000   # Initial size of replay memory before beginning network training

'''
This class represents the replay memory repository.
The main functions include storing new examples and returning sample sets
'''
class ReplayMemory(object):
    def __init__(self, init_size=INIT_SIZE, capacity=DEFAULT_CAPACITY):
        self.memory = {}
        self.first_index = -1
        self.last_index = -1
        self.capacity = capacity
        self.init_size = init_size

    # Store a new state, action, reward, next state (sars) tuple
    # If we have exceeded memory size, discard a sample too
    def store(self, sars_tuple):
        if self.first_index == -1:
            self.first_index = 0
        self.last_index += 1
        self.memory[self.last_index] = sars_tuple   
        if (self.last_index + 1 - self.first_index) > self.capacity:
            self.discard_sample()
    
    # True if we have reached the initial size
    def canTrain(self):
        return self.last_index + 1 - self.first_index >=self.init_size

    # True if we have reached capacity
    def is_full(self):
        return self.last_index + 1 - self.first_index >= self.capacity

    # True if replay memory is empty
    def is_empty(self):
        return self.first_index == -1

    # Discards oldest sample
    def discard_sample(self):
        rand_index = self.first_index
        del self.memory[rand_index]
        self.first_index += 1

    # Get a random sample
    def sample(self):
        if self.is_empty():
            raise Exception('Unable to sample from replay memory when empty')
        rand_sample_index = np.random.randint(self.first_index, self.last_index)
        return self.memory[rand_sample_index]

    # Get a set of samples. Samples are reorganized for batch training in the neural network
    def sample_batch(self,batch_size):
        # must insert data into replay memory before sampling
        if not self.canTrain():
            print 'CAN''T TRAIN YET!'
            print self.last_index+1-self.first_index
            print self.init_size
            return (-1,-1,-1,-1)
        if self.is_empty():
            raise Exception('Unable to sample from replay memory when empty')

        # determine shape of states
        state_shape = np.shape(self.memory.values()[0][0][0])
        states_shape = (batch_size,) + state_shape
        
        states1 = np.empty(states_shape)
        states2 = np.empty((batch_size,5))
        actions = np.empty((batch_size, 1))
        rewards = np.empty((batch_size, 1))
        next_states1 = np.empty(states_shape)
        next_states2 = np.empty((batch_size,5))
        
        # sample batch_size times from the memory
        for idx in range(batch_size):
            state, action, reward, next_state, = self.sample()
            states1[idx] = state[0]
            states2[idx] = state[1]
            actions[idx] = action
            rewards[idx] = reward
            
            next_states1[idx] = next_state[0]
            next_states2[idx] = next_state[1]
        return ([states1,states2],actions,rewards,[next_states1,next_states2])