import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import keras
import theano
import theano.tensor as T

TARGET_UPDATE_FREQ       = 1000    # Number of trainings before the target network is updated
SAVE_FREQ                = 1000    # Number of training iterations before the savining a new copy of the network
FINAL_EXPLORATION        = 0.1     # Final epsilon for epsilon greedy action selection
FINAL_EXPLORATION_FRAME  = 5e5     # Number of training samples where exploration is finished
BATCH_SIZE               = 2**5    # Batch size used to train network 
GAMMA                    = 0.99    # Discount factor
SOLVER                   = 'adamax'# Optimization method for neural network
NUM_ACTIONS = 2                    # Number of possible actions

ACTIONS = np.array([-5.0/180.0*np.pi, 5.0/180.0*np.pi]) # Possible changes in bank angle
SAVE_FILE = "./networks/TrainedNetwork_v1_%d.h5"  # Where network files should be written. Needs a %d at end for epoch number


''' 
Class representing the Q-network to train
The model uses the Keras library running on Theano
'''
class QNetwork(object):
    def __init__(self, init_size,save_file=SAVE_FILE, batch_size=BATCH_SIZE, num_actions=NUM_ACTIONS, discount=GAMMA, update_rule=SOLVER, rng=0):
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.discount = discount
        self.update_rule = update_rule
        self.save_file = save_file
        self.rng = rng if rng else np.random.RandomState()
        self.actions = ACTIONS
        self.saveFreq        = SAVE_FREQ
        self.replayStartSize = init_size
        self.finalExploration = FINAL_EXPLORATION
        self.finalExplorationSize = FINAL_EXPLORATION_FRAME
        self.targetNetworkUpdateFrequency = TARGET_UPDATE_FREQ
        self.initialize_network()
        self.update_counter = 0
        self.counter = -1.0
        
    # Get action
    # With probability epsilon, a random action is returned
    # Epsilon linearly decays from 1 to finalExploration over the initial training period until the replay memory reaches finalExplorationSize
    def getAction(self,inputs):
        belief,state = inputs
        self.counter+=1
        if self.counter < self.replayStartSize:
            return self.actions[self.rng.randint(self.num_actions)]     
        else:
            num = self.rng.rand()
            actInd=0
            if num>=np.min([(self.counter-self.replayStartSize),self.finalExplorationSize])/self.finalExplorationSize*(1-self.finalExploration):
                actInd = self.rng.randint(self.num_actions)
            else:
                actInd = np.argmax(self.model.predict([np.array([belief]),np.array([state])],batch_size=1, verbose = 0))
            return self.actions[actInd]
        
    # Defines the model and target architectures (same architecture, just two different models)
    def initialize_network(self):
        
        # Custom loss function
        # Squared error in places where error is not extremely negative
        # Error is extremely negative for actions that were not used
        # We only want to train on the action that was taken
        # The extremely negative error is a flag used in the Train method
        def both(y_true, y_pred):
            d = y_true-y_pred
            return T.switch(y_true>-50000.0,d**2,0)
                
        target1 = Sequential()
        target1.add(Convolution2D(64, 3, 3,init='zero', activation='relu',input_shape=(100,100,3)))
        target1.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
        target1.add(Convolution2D(64, 3, 3,init='zero', activation='relu'))
        target1.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
        target1.add(Convolution2D(64, 3, 3,init='zero', activation='relu'))
        target1.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
        target1.add(Flatten())
        target1.add(Dense(500,init='zero',activation='relu'))
        target1.add(Dense(100,init='zero',activation='relu'))

        target2 = Sequential()
        target2.add(Dense(100,input_dim=5,init='zero',activation='relu'))
        target2.add(Dense(100,input_shape=(1,),init='zero'))
        target2.add(Dense(100,activation='relu',init='zero'))
        target2.add(Dense(100,activation='relu',init='zero'))
        target2.add(Dense(100,activation='relu',init='zero'))
        targetMerged = Sequential()
        targetMerged.add(Merge([target1, target2], mode='concat', concat_axis=1))
        targetMerged.add(Dense(200, init='zero',activation='relu'))
        targetMerged.add(Dense(200, init='zero',activation='relu'))
        targetMerged.add(Dense(self.num_actions, init='zero'))
        targetMerged.compile(loss=both,optimizer=self.update_rule)

        model1 = Sequential()
        model1.add(Convolution2D(64, 3, 3,init='uniform', activation='relu',input_shape=(100,100,3)))
        model1.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
        model1.add(Convolution2D(64, 3, 3,init='uniform', activation='relu'))
        model1.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
        model1.add(Convolution2D(64, 3, 3,init='uniform', activation='relu'))
        model1.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
        model1.add(Flatten())
        model1.add(Dense(500,init='uniform',activation='relu'))
        model1.add(Dense(100,init='uniform',activation='relu'))

        model2 = Sequential()
        model2.add(Dense(100,input_dim=5,init='uniform',activation='relu'))
        model2.add(Dense(100,input_shape=(1,),init='uniform'))
        model2.add(Dense(100,activation='relu',init='uniform'))
        model2.add(Dense(100,activation='relu',init='uniform'))
        model2.add(Dense(100,activation='relu',init='uniform'))
        modelMerged = Sequential()
        modelMerged.add(Merge([model1, model2], mode='concat', concat_axis=1))
        modelMerged.add(Dense(200, init='uniform',activation='relu'))
        modelMerged.add(Dense(200, init='uniform',activation='relu'))
        modelMerged.add(Dense(self.num_actions, init='uniform'))
        modelMerged.compile(loss=both,optimizer=self.update_rule)
        
        self.target = targetMerged
        self.model = modelMerged
        
    # Train the network
    def train(self,(states,actions,rewards,next_states)):
        
        if self.update_counter % self.saveFreq ==0:
            self.saveModel()
        if self.update_counter % self.targetNetworkUpdateFrequency==0:
            print self.update_counter
            print "RESET TARGET NETWORK"
            self.reset_target_network()
        self.update_counter+=1
        
        # We want the value of the state to be the reward + discounted value of target network at next state
        # Compute the target network values at next states
        q_target = self.target.predict(next_states,batch_size = self.batch_size)
        
        # Initialize the desired values to be very negative. We only want to train with the action that was taken, so the
        # output corresponding to the taken action will have its value updated
        modelValues = q_target*0.0 - 100000.0
        for i in range(len(q_target)):
            # Set the value for the action index taken
            indTarget = np.argmax([q_target[i,:]])
            indModel  = int(actions[i]*18.0/np.pi+0.5)
            modelValues[i,indModel] = rewards[i]+self.discount*q_target[i,indTarget]
            
        # Train the model to approximate reward + disounted value of next state
        self.model.train_on_batch(states,modelValues)
    
    # Update the weights of the target network to be the model weights
    def reset_target_network(self):
        self.target.set_weights(self.model.get_weights())
    
    def getModel(self):
        return self.model
    def getTarget(self):
        return self.target
    
    # Save the model to file
    def saveModel(self):
        self.model.save((self.save_file % self.update_counter),overwrite=True)
        
    # Compute the loss for a batch of SARS
    def test(self,(states,actions,rewards,next_states)):
        q_target = self.target.predict(next_states,batch_size=self.batch_size)
        q_model  = self.model.predict(states,batch_size=self.batch_size)
        loss = 0.0
        for i in range(len(q_model)):
            modelInd = int(actions[i]*18.0/np.pi + 0.5)
            targetInd= np.argmax([q_target[i,:]])
            loss += (q_model[i,modelInd]-rewards[i]-self.discount*q_target[i,targetInd])**2
        return loss/len(q_target)
    
'''
Class for loading a trained model and evaluating it
'''
class QNetworkTrained(object):
    def __init__(self,filename):
        
        # Custom loss function
        def both(y_true, y_pred):
            d = y_true-y_pred
            return T.switch(y_true>-50000.0,d**2,0)
        
        self.filename = filename
        self.model = load_model(filename,custom_objects={'both':both})
        self.actions = ACTIONS
        
    # Return best action
    def getAction(self,inputs):
        scores = self.getActionCosts(inputs)
        actInd = np.argmax(scores)
        return self.actions[actInd]
    
    # Return costs of each action
    def getActionCosts(self,inputs):
        belief,state = inputs
        return self.model.predict([np.array([belief]),np.array([state])],batch_size=1, verbose = 0)