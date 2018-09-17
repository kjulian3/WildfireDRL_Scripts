import numpy as np
from AircraftModel import *
from Filters import *
from WildfireModels import *

## Wildfire aircraft settings
LAMDA1   = 5.0    # Tuning parameter for reward function
EKF_MODE = True   # Observation filter type, True for EKF, False for PF
NUM_UAVS = 2      # Number of UAVs to simulate
'''
Class representing the simulation environment
Has attributes for wildfire simulation, aircraft, and filters
'''
class Sim(object):
    def __init__(self,dt,dti,numUAVs = NUM_UAVS, ekf_mode=EKF_MODE,seed=None,windx=-2.,windy=-2.):
        self.fireModel = FireModel(seed=seed,windx=windx,windy=windy)
        self.numUAVs   = numUAVs
        self.aircraft  = [Aircraft(dt,dti) for i in range(self.numUAVs)] 
        self.ekf_mode=ekf_mode
        if not self.ekf_mode:
            self.PF = PF()
        self.ekf = independentEKF()
        self.reset()
        
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
  