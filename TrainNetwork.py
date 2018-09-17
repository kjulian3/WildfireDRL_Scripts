from math import sin, cos, pi, tan
import scipy
from scipy import eye, matrix, random, asarray, ndimage
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import keras

import theano
import theano.tensor as T
import math
import h5py
import random
import sys
import os         
           
    
## Set Parameters
DEFAULT_CAPACITY         = 70000   # Max size of replay memory
INIT_SIZE                = 35000   # Initial size of replay memory before beginning network training
TRAIN_FREQ               = 10      # Number of samples to generate between trainings
PRINT_FREQ               = 100     # Frequency of printing
TARGET_UPDATE_FREQ       = 1000    # Number of trainings before the target network is updated
SAVE_FREQ                = 1000    # Number of training iterations before the savining a new copy of the network
FINAL_EXPLORATION        = 0.1     # Final epsilon for epsilon greedy action selection
FINAL_EXPLORATION_FRAME  = 5e5     # Number of training samples where exploration is finished
BATCH_SIZE               = 2**5    # Batch size used to train network 
GAMMA                    = 0.99    # Discount factor
SOLVER                   = 'adamax'# Optimization method for neural network

## Wildfire aircraft settings
MINRANGE = 15   # Minimium initial distance from wildfire seed
MAXRANGE = 30   # Maximum initial distance from wildfire seed
XCENTER  = 50   # X-location of wildfire seed (2-97)
YCENTER  = 50   # Y-location of wildfire seed (2-97)
DT       = 0.5  # Time between wildfire updates               
DTI      = 0.1  # Time between aircraft decisions
NUM_ACTIONS = 2 # Number of possible actions
ACTIONS = np.array([-5.0/180.0*np.pi, 5.0/180.0*np.pi]) # Possible changes in bank angle
LAMDA1 = 5.0    # Tuning parameter for reward function
SAVE_FILE = "./networks/TrainedNetwork_v1_%d.h5"  # Where network files should be written. Needs a %d at end for epoch number

# Model settings
avgFuel = 18.0    # Average amount of fuel in cells
burnThresh = 22.  # Parameter for how quickly wildfire spreads. Higher number means wildfire spreads more slowly
alt=200.          # Altitude of aircraft 
errRate=0.1       # Error rate of observation images
ekf_mode = True   # Observation filter type, True for EKF, False for PF


'''
This class represents the aircraft dynamics and generates new states
'''
class StateGenerator(object):
    def __init__(self,minRange=MINRANGE,maxRange=MAXRANGE,xCenter=XCENTER,yCenter=YCENTER,seed=None):
        self.minRange  = minRange
        self.maxRange  = maxRange
        self.xCenter = xCenter
        self.yCenter = yCenter
        self.RNG = np.random.RandomState(seed)
        
    def setRandomSeed(self,seed):
        self.RNG = np.random.RandomState(seed)
        
    '''
    The aircraft dynamics.
    The state has four components:
        x-position
        y-position
        th- heading angle of aircraft
        phi- bank angle of aircraft
    '''
    def getNextState(self,state,action,dt=DT,dti=DTI,vown=20):  
        x,y,th,phi = state
        i=0
        
        # Whild wildfire sim coordinates range from 0-100, each wildfire coordinate is a 10m square
        # To update position, first convert to position in meters
        x*=10
        y*=10
        
        # Calculate position and heading angle
        while i<dt:
            i+=dti
            
            x += vown*cos(th)*dti
            y += vown*sin(th)*dti
            th += 9.80*math.tan(phi)/vown*dti

            if (th>math.pi):
                th-=2*math.pi
            elif (th<-math.pi):
                th+=2*math.pi
                
        # Update bank angle and convert position back wildfire coordinates
        phi += action
        x/=10
        y/=10
        
        # Limit bank angle 
        if phi > 50.0*np.pi/180.0 or phi < -50.0*np.pi/180.0:
            phi -= action
        return (x,y,th,phi)
    
    '''
    Get a new state. Initial bank angle is assumed 0
    '''
    def randomStateGenerator(self):
        r = self.RNG.rand()*(self.maxRange-self.minRange) + self.minRange
        theta = (self.RNG.rand()-0.5)*2*np.pi
        x = r*np.cos(theta) + self.xCenter
        y = r*np.sin(theta) + self.yCenter
        th = (self.RNG.rand()-0.5)*2*np.pi
        phi = 0
        return (x,y,th,phi)
    
'''
Computes position and orientation of an aircraft's state relative to another
Used as input to network
'''
def getRelativeState(state1,state2):
    x,y,th,phi     = state1
    x2,y2,th2,phi2 = state2

    r = np.sqrt((x-x2)**2 + (y-y2)**2)
    theta = np.arctan2(y2-y,x2-x)-th
    psi   = th2 - th

    if (theta > math.pi):
        theta -= 2*math.pi
    elif (theta<-math.pi):
        theta+= 2*math.pi

    if (psi > math.pi):
        psi -= 2*math.pi
    elif (psi<-math.pi):
        psi+= 2*math.pi
    return (phi,r,theta,psi,phi2)

'''
This class represents the replay memory repository.
The main functions include storing new examples and returning sample sets
'''
class ReplayMemory(object):
    def __init__(self, batch_size=BATCH_SIZE, init_size=INIT_SIZE, capacity=DEFAULT_CAPACITY):
        self.memory = {}
        self.batch_size = batch_size
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

    # Discards a random sample
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
    def sample_batch(self):
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
        states_shape = (self.batch_size,) + state_shape
        
        states1 = np.empty(states_shape)
        states2 = np.empty((self.batch_size,5))
        actions = np.empty((self.batch_size, 1))
        rewards = np.empty((self.batch_size, 1))
        next_states1 = np.empty(states_shape)
        next_states2 = np.empty((self.batch_size,5))
        
        # sample batch_size times from the memory
        for idx in range(self.batch_size):
            state, action, reward, next_state, = self.sample()
            states1[idx] = state[0]
            states2[idx] = state[1]
            actions[idx] = action
            rewards[idx] = reward
            
            next_states1[idx] = next_state[0]
            next_states2[idx] = next_state[1]
        return ([states1,states2],actions,rewards,[next_states1,next_states2])
    
''' 
Class representing the Q-network to train
The model uses the Keras library running on Theano
'''
class QNetwork(object):
    def __init__(self, save_file=SAVE_FILE, batch_size=BATCH_SIZE, num_actions=NUM_ACTIONS, discount=GAMMA, update_rule=SOLVER, rng=0, init_size=INIT_SIZE):
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
    
'''
A class representing an aircraft and its sensors
'''
class Aircraft(object):
    def __init__(self,sg,alt=150.,errRate=0.1,seed=None):
        self.sg = sg
        self.obsModel = ObservationModel(alt=alt,errRate=errRate)
        if seed is not None:
            self.seed=seed-1               
        else:
            self.seed=None
    
    # Get sensor information
    def getSensor(self,burnMap):
        return self.obsModel.fastObservation(self.state,burnMap)
    
    # Increment aircraft state given action
    def step(self,action):
        self.state=self.sg.getNextState(self.state,action)
       
    # Reset the position and observation model of aircraft
    def reset(self):
        if self.seed is not None:
            self.seed+=1
            
        self.sg.setRandomSeed(self.seed)
        self.obsModel.setRandomSeed(self.seed)
        self.state = self.sg.randomStateGenerator()
        
'''
Class representing the simulation environment
Has attributes for wildfire simulation, aircraft, and filters
'''
class Sim(object):
    def __init__(self,avgFuel,bt,aircraft,ekf_mode=False,seed=None,windx=-2.,windy=-2.):
        self.fireModel = FireModel(avgFuel=avgFuel,bt=bt,seed=seed,windx=windx,windy=windy)
        self.aircraft = aircraft
        self.ekf_mode=ekf_mode
        self.numUAVs = len(aircraft)
        if not self.ekf_mode:
            self.PF = PF()
            self.PF_resample_counter=-10

        self.ekf = independentEKF()
        self.ekf.reset(self.fireModel.startLocation1)
        
    '''
    Reset fire model, aircraft, and filters
    '''
    def reset(self):
        self.fireModel.reset()
        [ac.reset() for ac in self.aircraft]
        self.ekf.reset(self.fireModel.startLocation1)
        if not self.ekf_mode:
            self.PF.reset()
            self.PF_resample_counter=-10
            self.seenFireFlag = False
       
    '''
    Step the fires forward
    '''
    def stepFires(self):
        self.fireModel.step()
        if not self.ekf_mode:
            self.PF.step()
            self.PF_resample_counter+=1
        
    '''
    Step the aircraft forward given a list of actions
    '''
    def stepAircraft(self,actions):
        for ac, action in zip(self.aircraft,actions):
            ac.step(action)
       
    '''
    Check if all of the fire is within the allowed range
    '''
    def fireInRange(self,margin=2):
        burnX,burnY = np.where(self.fireModel.burnMap.reshape((100,100))==1)
        return min(burnX)>=margin and min(burnY)>=margin and max(burnX)<=99-margin and max(burnY)<=99-margin
    
    '''
    Updates the filters with observations and computes the reward
    '''
    def updateEKFandReward(self):
        self.ekf.stepTimeMap()
        rewards = []
        
        if not self.ekf_mode and self.PF_resample_counter==20:
            self.PF.resample()
            self.PF_resample_counter=0
                
        for acInd, ac in enumerate(self.aircraft):
            
            ## Update EKF given sensor information
            sensors = ac.getSensor(self.fireModel.burnMap.reshape((100,100)))
            self.ekf.update(sensors)
            
            if not self.ekf_mode:
                self.PF.update(sensors)
                if not self.seenFireFlag and self.PF_resample_counter<0 and np.sum(self.ekf.seenWildfire)>180:
                    self.PF_resample_counter=0
                    self.seenFireFlag=True
            
            # Get Rewards
            rewards+= [self.getReward(acInd)]
        return rewards
              
    '''
    Compute the reward for an aircraft given state of wildfire and its position
    The EKF tracks wildfire that has been seen, used to penalize aircraft for flying over seen wildfire
    '''
    def getReward(self,acInd):
        # Penalty for being too close to wildfire
        x,y,_,_ = self.aircraft[acInd].state
        wildfirePenalty = 0
        for i in range(int(x)-4,int(x)+5):
            for j in range(int(y)-4,int(y)+5):
                if i>=0 and i<100 and j>=0 and j<100 and (x-i)**2+(y-j)<=16:
                    wildfirePenalty -= LAMDA1 * self.ekf.seenWildfire[j,i]
        
        
        #Penalty for closeness to intruder
        relState = getRelativeState(self.aircraft[0].state,self.aircraft[1].state)
        r = relState[1]
        proximityPenalty = -1.5*np.exp(-r/10.0)
        
        
        # Reward for seeing new wildfire
        currentWildfireThresh = self.ekf.thresholdMu()
        wildfireReward = np.sum((currentWildfireThresh==1) & (self.ekf.seenWildfire==0))
        self.ekf.seenWildfire[currentWildfireThresh==1]=1
        
        return proximityPenalty + wildfireReward + wildfirePenalty

    '''
    This function computes the inputs to the network
    The relative belief map, time map, and seen maps are concatenated to form a 3-channel image, and the relative state of the other aircraft is also used
    '''
    def getBeliefAndState(self, acInd,acInd2):
        x,y,th,phi = self.aircraft[acInd].state
        
        rMat = np.sqrt((self.fireModel.gridX.reshape((100,100))-50)**2+(self.fireModel.gridY.reshape((100,100))-50)**2)
        thMat = np.arctan2(self.fireModel.gridY.reshape((100,100))-50,self.fireModel.gridX.reshape((100,100))-50)
        xRel = np.round(x+rMat*np.cos(th+thMat)).astype(int)
        yRel = np.round(y+rMat*np.sin(th+thMat)).astype(int)
        
        xRel2 = np.where((xRel>=0) & (xRel < 100),xRel,0)
        yRel2 = np.where((yRel>=0) & (yRel < 100),yRel,0)
        
        xRel3 = np.where((xRel>=-100) & (xRel < 200),xRel,0)
        yRel3 = np.where((yRel>=-100) & (yRel < 200),yRel,0)
           
        relTimeMap   = np.where((xRel>-100) & (xRel<200) & (yRel>-100) & (yRel<200),self.ekf.timeMap[yRel3+100,xRel3+100],1)
        
        if self.ekf_mode:
            relBeliefMap = np.where((xRel>=0) & (xRel<100) & (yRel>=0) & (yRel<100),self.ekf.mu[yRel2,xRel2],0)
            relBeliefMapSeen = np.where((xRel>=0) & (xRel<100) & (yRel>=0) & (yRel<100),self.ekf.seenWildfire[yRel2,xRel2],0)
        else:
            burnThresh = self.PF.getMeanBurnThresh().reshape((100,100))
            burnedMap = self.PF.getBurned().reshape((100,100))
            relBeliefMap = np.where((xRel>=0) & (xRel<100) & (yRel>=0) & (yRel<100),burnThresh[yRel2,xRel2],0)
            relBeliefMapSeen = np.where((xRel>=0) & (xRel<100) & (yRel>=0) & (yRel<100),burnedMap[yRel2,xRel2],0)
                    
        relState = getRelativeState(self.aircraft[acInd].state,self.aircraft[acInd2].state)
        return [np.stack([relBeliefMap,relTimeMap,relBeliefMapSeen],axis=2), relState]

    '''
    Computes vertices of sensor region on ground using a Convex Hull
    '''
    def getSensorVerts(self):
        sensorVerts = {}
        for acInd, ac in enumerate(self.aircraft):
            sensorVerts[acInd] = []
            sensorTuples = ac.getSensor(self.fireModel.burnMap.reshape((100,100)))
            for sensor in sensorTuples:
                points, obs = sensor
                inds = np.where((points[0]>0) & (points[0]<99) & (points[1]>0) & (points[1]<99))[0]
                try:
                    hull = ConvexHull(points[:,inds].T)
                    if hull is not None:
                        vert = np.concatenate((hull.vertices,np.array([hull.vertices[0]])))
                        sensorVerts[acInd] += [hull.points[vert]]
                except:
                    sensorVerts[acInd] += [np.array([])]
        return sensorVerts
  
'''
Rotation matrix calculations
'''
def rot(angle,coord):
    if coord==1:
        return np.array([[1,0,0],[0,math.cos(angle),-math.sin(angle)],[0,math.sin(angle),math.cos(angle)]])
    if coord==2:
        return np.array([[math.cos(angle),0,math.sin(angle)],[0,1,0],[-math.sin(angle),0,math.cos(angle)]])
    return np.array([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]])

'''
Class modeling the cameras on an aircraft
'''
class ObservationModel(object):
    def __init__(self,alt=200., errRate = 0.1,phis=[40./180*np.pi,13./180*np.pi,-13./180*np.pi,-40./180*np.pi],theta=30./180.*np.pi,f=50., imX=24., imY=36.0, pixX=20, pixY=30, rnge=300.):
        self.errRate = errRate
        self.phis = phis
        self.theta=theta
        self.f = f
        self.imX = imX
        self.imY = imY
        self.pixX = pixX
        self.pixY = pixY
        self.range = rnge
        self.rotMats = [rot(self.theta,2).dot(rot(self.phis[i],1)) for i in range(len(phis))]

        self.minX = -imX/2/f
        self.minY = -imY/2/f
        self.maxX = imX/2/f
        self.maxY = imY/2/f
        self.xxx, self.yyy = np.meshgrid(np.linspace(-100,199,300), np.linspace(-100,199,300))
        self.yyy = self.yyy.reshape(-1)
        self.xxx = self.xxx.reshape(-1)
        self.alt=alt
        
    def setRandomSeed(self,seed):
        self.RNG = np.random.RandomState(seed)
    
    def fastObservation(self,state, burnMap):
        x,y,th,bank = state
        f = self.f
        imX = self.imX
        imY = self.imY
        pixX = self.pixX
        pixY = self.pixY
        theta = self.theta
        
        xxx = self.xxx
        yyy = self.yyy
        
        returnTuples = []
        for phi, rotMat in zip(self.phis,self.rotMats):
            
            # Compute points on ground from image coordinates
            R_b_c = rot(bank,1).dot(rotMat)
            
            xxx_Possible = xxx[(xxx-x)**2+(yyy-y)**2<self.range**2/100.]
            yyy_Possible = yyy[(xxx-x)**2+(yyy-y)**2<self.range**2/100.]
            
            xRel = 10*(xxx_Possible-x)*cos(th)+10*(yyy_Possible-y)*sin(th)
            yRel = 10*(yyy_Possible-y)*cos(th)-10*(xxx_Possible-x)*sin(th)
            zRel = self.alt*np.ones(len(xxx_Possible))
            points2 = (R_b_c.T.dot([xRel,yRel,zRel]))
            
            # Round to nearest point in image
            points2yy = (np.round((points2[0]/points2[2]/self.maxY+1)/2*(self.pixY-1))/(self.pixY-1)*2-1)*self.maxY
            points2xx = (np.round((points2[1]/points2[2]/self.maxX+1)/2*(self.pixX-1))/(self.pixX-1)*2-1)*self.maxX
            
            # Get the good points
            inds = (points2[2]>0) & (abs(points2yy)<=self.maxY) & (abs(points2xx)<=self.maxX)
            yyy_good = yyy_Possible[inds]
            xxx_good = xxx_Possible[inds]
            imPointx_good = points2xx[inds]
            imPointy_good = points2yy[inds]
            
            # Map image points to point on ground
            x3 = self.alt/(imPointy_good*(-cos(bank)*sin(theta))+imPointx_good*(cos(phi)*sin(bank) + cos(bank)*cos(theta)*sin(phi))+ cos(bank)*cos(phi)*cos(theta) - sin(bank)*sin(phi))
            x1 = imPointy_good*x3
            x2 = imPointx_good*x3
            points = (R_b_c.dot([x1,x2,x3]))
            xRel = np.round(x+points[0]/10.*math.cos(th)-points[1]/10.0*math.sin(th)).astype(int)
            yRel = np.round(y+points[0]/10.*math.sin(th)+points[1]/10.0*math.cos(th)).astype(int)
            
            # Obtain observation
            xRel2 = np.where((xRel>=0) & (xRel < 100),xRel,0)
            yRel2 = np.where((yRel>=0) & (yRel < 100),yRel,0)
            obs = np.where((xRel>=0) & (xRel<100) & (yRel>=0) & (yRel<100),burnMap[yRel2,xRel2],0)
            obs = np.where((xRel>=0) & (xRel<100) & (yRel>=0) & (yRel<100) & (self.RNG.rand(len(obs))<self.errRate),1-obs,obs)

            goodPoints = np.array([xxx_good,yyy_good]).astype(int)
            returnTuples += [(goodPoints, obs)]

        return returnTuples

'''
Simplified independent EKF filter
'''
class independentEKF:
    def __init__(self):
        
        gridPoints = np.zeros((300*300,2),int)
        for i in range(300):
            for j in range(300):
                gridPoints[300*i+j,:] = [i-100,j-100]
                
        self.gridPoints=gridPoints

        self.Q = 0.01
        self.R = 0.25
        self.mu = np.zeros((100,100))
        self.sigma = np.ones((100,100))*0.01
        self.timeMap = np.ones((300,300))
        self.seenWildfire = np.zeros(self.mu.shape)
    
    def thresholdMu(self):
        thresh = np.zeros(self.mu.shape)
        thresh[np.where(self.mu>0.6)]=1.0
        return thresh
    
    def stepTimeMap(self):
        self.timeMap+=1.0/255.0
        self.timeMap[self.timeMap>1.0] = 1.0
        
        
    def reset(self,startLocation1):
        self.sigma = np.ones((100,100))*0.01
        self.mu = np.zeros((100,100))
        self.timeMap = np.ones((300,300))
        self.seenWildfire = np.zeros(self.mu.shape)
        for i in range(-2,3):
            for j in range(-2,3):
                self.mu[startLocation1[0]+i,startLocation1[1]+j] = 1
                
        for i in range(-6,7):
            for j in range(-6,7):
                self.seenWildfire[startLocation1[0]+i,startLocation1[1]+j] = 1
        
    '''
    Assume each cell is independent, so this is like having many 1-D EKF's
    '''
    def update(self,sensors):
        visited = np.zeros((300,300)).astype(int)
        
        for sensor in sensors:
            points,obs = sensor
            
            inds = visited[points[1]+100,points[0]+100]==0
            points = points[:,inds]
            obs = obs[inds]
            visited[points[1]+100,points[0]+100] += 1
            
            
            self.timeMap[points[1]+100,points[0]+100] = 0.0
            
            inds = np.where((points[0]>=0) & (points[0]<100) & (points[1]>=0) & (points[1]<100))[0]
            points = points[:,inds]
            obs = obs[inds]

            self.sigma[points[1],points[0]]+=self.Q
            K = self.sigma[points[1],points[0]]/(self.sigma[points[1],points[0]]+self.R)

            self.mu[points[1],points[0]] +=  K*(obs-self.mu[points[1],points[0]])
            self.sigma[points[1],points[0]]-=K*self.sigma[points[1],points[0]]

            self.mu[points[1],points[0]] = np.where(self.mu[points[1],points[0]]>1.0,1.0,self.mu[points[1],points[0]])
            self.mu[points[1],points[0]] = np.where(self.mu[points[1],points[0]]<0.0,0.0,self.mu[points[1],points[0]])
 
'''
Complex particle filter
'''
class PF(object):
    def __init__(self, seed=None, includeArc=False, numPart=40,Qroot=np.diag([0.001,0.001])):
        
        self.numPart =numPart
        self.Qroot = Qroot
        self.includeArc=includeArc
        
        self.PF = {}
        self.PF_noUpdate = {}
        self.probCorrectObs = 0.8
        self.seed = seed
        if self.seed is not None:
            self.seed-=1
        
    def update(self,weights,state):
        self.step()
        self.resample(weights,state)
    
    def getParticles(self):
        return self.PF            
                        
    def step(self):
        for ii in range(self.numPart):
            self.PF[ii].step()
            self.PF_noUpdate[ii].step()
                             
            deltaWind = self.RNG.multivariate_normal(np.zeros(2),self.Qroot,1)[0]
            winds = [self.PF[ii].windx+deltaWind[0], self.PF[ii].windy+deltaWind[1]]
            for i in [0,1]:
                if winds[i]<-1.0:
                    winds[i]=-1.0
                elif winds[i]>1.0:
                    winds[i] = 1.0
            
            self.PF[ii].windx = winds[0]
            self.PF[ii].windy = winds[1]
            self.PF_noUpdate[ii].windx = winds[0]
            self.PF_noUpdate[ii].windy = winds[1]
            
     
    def normalizeWeights(self):
        weights = self.weights.copy()
        weights-= np.max(weights)
        ind = np.argsort(weights)[-8]
        if weights[ind]<0:
            weights/= abs(weights[ind]/3)
        weights = np.exp(weights)
        weights/=np.sum(weights)
        return weights
    
    def resample(self):
        
        weights = self.normalizeWeights()
        
        inds = [self.RNG.choice(range(self.numPart),p=weights) for _ in range(self.numPart)]
        newPF = {}
        newPF_noUpdate = {}
        for i in range(len(inds)):
            newPF[i] = self.PF[inds[i]].copy()
            newPF_noUpdate[i] = self.PF[inds[i]].copy()
        self.PF = newPF
        self.PF_noUpdate = newPF_noUpdate
        self.weights = np.zeros(self.numPart)
        
    def reset(self):
        if self.seed is not None:
            self.seed+=1
            
        self.RNG = np.random.RandomState(self.seed)
        self.weights = np.zeros(self.numPart)
        for ii in range(self.numPart):
            winds = self.RNG.rand(2)*2-1
            self.PF[ii] = FireModel_Probs(windx=winds[0],windy=winds[1],includeArc=(self.includeArc and ii>self.numPart/2))
            self.PF_noUpdate[ii] = self.PF[ii].copy()
            
    def getEstimates(self):
        est = []
        for key in self.PF.keys():
            est.append([self.PF[key].windx,self.PF[key].windy])
        return np.array(est)
    
    def getWeights(self):
        if min(self.weights)==max(self.weights):
            return np.ones(self.numPart)/self.numPart
        return self.normalizeWeights()
    
    def estimateWind(self):
        winds = np.sum(self.getWeights().reshape((self.numPart,1))*self.getEstimates(),axis=0)
        return winds
    
    def plotParticle(self, ind):
        plotter = fireSimPlotter_Fire(self.PF[ind])
        return plotter.plot()
    
    def getMeanImages(self,ind=-1):
        if ind==-1:
            burnMean = np.sum([self.PF[p].burnMapProbs for p in self.PF],axis=0)/float(self.numPart)
            fuelMean = np.sum([np.sum([i*self.PF[p].fuelMapProbsList[i] for i in range(len(self.PF[p].fuelMapProbsList))],axis=0) for p in self.PF],axis=0)/float(self.numPart)
        else:
            burnMean = self.PF[ind].burnMapProbs
            fuelMean = np.sum([i*self.PF[ind].fuelMapProbsList[i] for i in range(len(self.PF[ind].fuelMapProbsList))],axis=0)
        return burnMean, fuelMean
    
    def getMeanBurn(self):
        weights = self.getWeights()
        burnMean = np.sum([weights[p]*self.PF[p].burnMapProbs for p in self.PF],axis=0)
        return burnMean
    
    def getMeanFuel(self):
        return np.sum([np.sum([i*self.PF[p].fuelMapProbsList[i] for i in range(len(self.PF[p].fuelMapProbsList))],axis=0)*self.getWeights()[p] for p in self.PF],axis=0)
    
    def getMeanBurnThresh(self):
        burnMean = self.getMeanBurn()
        return np.where(burnMean>0.5,1,0)
    
    def getBurned(self):
        meanBurn = self.getMeanBurn()
        meanFuel = self.getMeanFuel()
        return np.where((meanBurn>0.5) | (meanFuel<10.),1,0)
    
    def update(self,sensors):
        for sensor in sensors:
            points, obs = sensor
            inds = np.where((points[0]>=0) & (points[0]<=99) & (points[1]>=0) & (points[1]<=99))[0]
            self.updateWeights(points[:,inds],obs[inds])
            self.updateBelief(points[:,inds],obs[inds])
    
    def updateWeights(self,points,obs):
        eps = 2e-4
        for i in range(self.numPart):
            bmp = self.PF_noUpdate[i].burnMapProbs.reshape((100,100))
            self.weights[i] += np.sum(np.log(eps+bmp[points[1,obs==1],points[0,obs==1]]))
            self.weights[i] += np.sum(np.log(eps+1-bmp[points[1,obs==0],points[0,obs==0]]))

    def updateBelief(self, points, obs):
        for i in range(self.numPart):
            bmp = self.PF[i].burnMapProbs.reshape((100,100))
            bmp[points[1,obs==1],points[0,obs==1]]=self.probCorrectObs*bmp[points[1,obs==1],points[0,obs==1]]/(self.probCorrectObs*bmp[points[1,obs==1],points[0,obs==1]] + (1-self.probCorrectObs)*(1-bmp[points[1,obs==1],points[0,obs==1]]))
            bmp[points[1,obs==0],points[0,obs==0]]=(1-self.probCorrectObs)*bmp[points[1,obs==0],points[0,obs==0]]/((1-self.probCorrectObs)*bmp[points[1,obs==0],points[0,obs==0]] + self.probCorrectObs*(1-bmp[points[1,obs==0],points[0,obs==0]]))

'''
Wildfire model
'''
class FireModel(object):
    def __init__(self,avgFuel=18.0,bt=22.0,includeArc=False,seed=None,windx=-2.0,windy=-2.0,burnMap=None,fuelMap=None):
        
        if seed is not None:
            self.seed=seed-1                
        else:
            self.seed=None
            
        gridPoints2 = np.zeros((300*300,2),int)
        for i in range(300):
            for j in range(300):
                gridPoints2[300*i+j,:] = [i-100,j-100]
        
        self.minFuel  = avgFuel-3
        self.maxFuel  = avgFuel+3
        self.burnRate = 1.0
        
        gridX     = np.linspace(0,99,100)
        gridY     = np.linspace(0,99,100)
        self.startLocation1 = (gridY.size/2,gridX.size/2)
        self.size = gridX.size
        self.gridX,self.gridY =  np.meshgrid(gridX,gridY)
        self.gridX = self.gridX.reshape(-1).astype(int)
        self.gridY = self.gridY.reshape(-1).astype(int)
        self.burnThresh = bt
        self.includeArc = includeArc
        
        self.windx_given=windx
        self.windy_given=windy
        
        if burnMap is not None and fuelMap is not None  and windy != -2 and windx !=-2:
            self.burnMap = burnMap
            self.fuelMap = fuelMap
            self.windx = windx
            self.windy = windy
            self.WildfireRNG = np.random.RandomState(None)
    
    def step(self):
        self.fuelMap = np.where(self.burnMap==1, self.fuelMap-self.burnRate,self.fuelMap)
        self.fuelMap = np.where(self.fuelMap<0,0,self.fuelMap)

        i = self.gridX
        j = self.gridY

        probsMap = np.zeros(len(self.burnMap))
        for ii in [-2,-1,0,1,2]:
            for jj in [-2,-1,0,1,2]:
                if not (ii ==0 and jj==0):
                    inds = (i>=-ii) & (i<100-ii) & (j>=-jj) & (j<100-jj)
                    probsMap[i[inds]*100+j[inds]] += max(0,1.0-np.sign(ii)*self.windy-np.sign(jj)*self.windx)/(ii*ii + jj*jj)*self.burnMap[(ii+i[inds])*100+jj+j[inds]]

        self.burnMap = np.where( (self.fuelMap>0) & ((self.burnMap==1) | (probsMap>self.WildfireRNG.rand(10000)*self.burnThresh)),1,0)
    
    def reset(self):
        if self.seed is not None:
            self.seed+=1
        self.WildfireRNG = np.random.RandomState(self.seed)
        self.fuelMap = self.WildfireRNG.randint(self.minFuel,self.maxFuel,self.gridX.shape)
        self.burnMap = np.zeros((100,100))

        for i in [-2,-1,0,1,2]:
            for j in [-2,-1,0,1,2]:
                self.burnMap[self.startLocation1[0]+i,self.startLocation1[1]+j] = 1
        self.burnMap = self.burnMap.reshape(-1)
                
                
        if self.windx_given==-2 and self.windy_given==-2:
            self.windx = (self.WildfireRNG.rand()-0.5)*1.6
            self.windy = (self.WildfireRNG.rand()-0.5)*1.6
        else:
            self.windx=self.windx_given
            self.windy=self.windy_given
        
        if self.includeArc and self.WildfireRNG.randint(2)==1:
            self.fuelMap[52:,:] = 0
        for i in range(30):
            self.step()
            
    def copy(self):
        return FireModel(windx=self.windx,windy=self.windy,fireMap=self.fuelMap.copy(),burnMap=self.burnMap.copy())
   
'''
Probabilistic wildfire model, used by particle filter
'''
class FireModel_Probs(object):
    def __init__(self,windx,windy,avgFuel=18.0,bt=22.0,includeArc=False,burnMapProbs=None,fuelMapProbsList=None):
            
        gridPoints2 = np.zeros((300*300,2),int)
        for i in range(300):
            for j in range(300):
                gridPoints2[300*i+j,:] = [i-100,j-100]
        
        self.minFuel  = avgFuel-3
        self.maxFuel  = avgFuel+3
        self.burnRate = 1.0
        
        gridX     = np.linspace(0,99,100)
        gridY     = np.linspace(0,99,100)
        self.startLocation1 = (gridY.size/2,gridX.size/2)
        self.size = gridX.size
        self.gridX,self.gridY =  np.meshgrid(gridX,gridY)
        self.burnThresh = bt
        self.includeArc = includeArc
        
        self.windx=windx
        self.windy=windy
        
        if burnMapProbs is not None and fuelMapProbsList is not None and windy != -2 and windx !=-2:
            self.burnMapProbs = burnMapProbs
            self.fuelMapProbsList = fuelMapProbsList
            self.windx = windx
            self.windy = windy
        else:
            self.reset()
    
    def step(self):
        self.fuelMapProbsList[0] = self.fuelMapProbsList[0] + self.fuelMapProbsList[1]*self.burnMapProbs
        for i in range(1,int(self.maxFuel)):
            self.fuelMapProbsList[i] = self.fuelMapProbsList[i]*(1-self.burnMapProbs)+self.fuelMapProbsList[i+1]*self.burnMapProbs    
        
        i = self.gridX.reshape(-1).astype(int)
        j = self.gridY.reshape(-1).astype(int)

        pIgnite = np.zeros(len(self.burnMapProbs))
        for ii in [-2,-1,0,1,2]:
            for jj in [-2,-1,0,1,2]:
                if not (ii ==0 and jj==0):
                    inds = (i>=-ii) & (i<100-ii) & (j>=-jj) & (j<100-jj)
                    pIgnite[i[inds]*100+j[inds]] += max(0,1.0-np.sign(ii)*self.windy-np.sign(jj)*self.windx)/(ii*ii + jj*jj)*self.burnMapProbs[(ii+i[inds])*100+jj+j[inds]]/(self.burnThresh+3)

        pIgnite[pIgnite>1.0]=1.0
        self.burnMapProbs = (1-self.fuelMapProbsList[0])*((1-self.burnMapProbs)*pIgnite + self.burnMapProbs) #-self.fuelMapProbsList[0]* self.burnMapProbs
    
    def reset(self):
        self.burnMapProbs = np.zeros(self.gridX.shape)
        self.fuelMapProbsList  = [np.zeros(self.size**2) for _ in range(int(self.minFuel))] 
        self.fuelMapProbsList += [np.ones(self.size**2)/(self.maxFuel-self.minFuel) for _ in range(int(self.minFuel),int(self.maxFuel))]
        self.fuelMapProbsList += [np.zeros(self.size**2)]
        
        for i in [-2,-1,0,1,2]:
            for j in [-2,-1,0,1,2]:
                self.burnMapProbs[self.startLocation1[0]+i,self.startLocation1[1]+j] = 1
        self.burnMapProbs = self.burnMapProbs.reshape(-1)
         
        if self.includeArc==1:
            self.fuelMap[52:,:] = 0
            
        for i in range(30):
            self.step()
            
    def copy(self):
        return FireModel_Probs(self.windx,self.windy,fuelMapProbsList=[f.copy() for f in self.fuelMapProbsList],burnMapProbs=self.burnMapProbs.copy())
    
    
''' Train Network '''
# Get state generator, QNetwork, and Replay Memory, Aircraft, and Simulation objects
sg = StateGenerator()
q = QNetwork()
repMem = ReplayMemory()
aircraft = [Aircraft(sg,alt=alt,errRate=errRate) for i in range(2)] 
sim = Sim(avgFuel,burnThresh,aircraft,ekf_mode=ekf_mode)
sim.reset()

# Loop forever 
count=0
while True:
    
    # Number of loops before updating network
    for j in range(TRAIN_FREQ/np.int(2*DT/DTI)):
        # Step wildfires
        sim.stepFires()
        state0 = sim.getBeliefAndState(0,1); state1 = sim.getBeliefAndState(1,0);
        
        #Update aircraft and store state-action-reward-next state tuples in replay memory
        for i in range(np.int(DT/DTI)):
            action0 = q.getAction(state0); action1 = q.getAction(state1)                         # Compute actions
            sim.stepAircraft([action0,action1])                                                  # Update aircraft positions
            rewards = sim.updateEKFandReward()                                                   # Update sensors and compute reward
            nextState0 = sim.getBeliefAndState(0,1); nextState1 = sim.getBeliefAndState(1,0);    # Compute the relative belief and state
            repMem.store((state0,action0,rewards[0],nextState0))                                 # Store SARS tuple for first aircraft
            repMem.store((state1,action1,rewards[1],nextState1))                                 # Store SARS tuple for second aircraft
            count+=2                                                                             # Update counter by two since we stored two samples
            state0 = nextState0; state1 = nextState1;                                            # Update state to be next state
               
        # Reset wildfire simulation when wildfire gets large
        if not sim.fireInRange(6):
            sim.reset()
            state0 = sim.getBeliefAndState(0,1); state1 = sim.getBeliefAndState(1,0);
            
    # Printing
    if (count % PRINT_FREQ) ==0 and count>=INIT_SIZE:
        print "Samples: %d, Trainings: %d" % (count,(count-INIT_SIZE)/TRAIN_FREQ),"Loss: %.3e" % q.test(repMem.sample_batch())
        sys.stdout.flush()
    elif (count % 1000) ==0 and count<INIT_SIZE:
        print "Samples: %d" % count
        sys.stdout.flush()
        
    # Train network on a batch from replay memory
    if count>=INIT_SIZE:
        q.train(repMem.sample_batch())