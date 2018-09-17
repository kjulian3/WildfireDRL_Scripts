import numpy as np
from math import sin, cos, pi, tan
import math

# Aircraft position and update parameters
MINRANGE = 15   # Minimium initial distance from wildfire seed
MAXRANGE = 30   # Maximum initial distance from wildfire seed
XCENTER  = 50   # X-location of wildfire seed (2 - 97)
YCENTER  = 50   # Y-location of wildfire seed (2 - 97)

# Sensor Paratmerss
ALT        = 200.      # Altitude of aircraft 
ERROR_RATE = 0.1       # Error rate of observation images
PHIS       = [40./180*np.pi,13./180*np.pi,-13./180*np.pi,-40./180*np.pi] # Rotation angle of camera around aircraft body x-axis
THETA      = 30./180*np.pi # Subsequent rotation angle of camera around camera y-axis
FOCAL_LENGTH = 50. # Focal length of camera (mm)
'''
This class represents the aircraft dynamics and generates new states
'''
class StateGenerator(object):
    def __init__(self,dt,dti,minRange=MINRANGE,maxRange=MAXRANGE,xCenter=XCENTER,yCenter=YCENTER,seed=None):
        self.minRange  = minRange
        self.maxRange  = maxRange
        self.xCenter = xCenter
        self.yCenter = yCenter
        self.RNG = np.random.RandomState(seed)
        self.dt = dt
        self.dti = dti
        
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
    def getNextState(self,state,action,vown=20):  
        x,y,th,phi = state
        i=0
        
        # Whild wildfire sim coordinates range from 0-100, each wildfire coordinate is a 10m square
        # To update position, first convert to position in meters
        x*=10
        y*=10
        
        # Calculate position and heading angle
        while i<self.dt:
            i+=self.dti
            
            x += vown*cos(th)*self.dti
            y += vown*sin(th)*self.dti
            th += 9.80*math.tan(phi)/vown*self.dti

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
A class representing an aircraft and its sensors
'''
class Aircraft(object):
    def __init__(self,dt,dti,alt=ALT,errRate=ERROR_RATE,seed=None):
        self.sg = StateGenerator(dt,dti)
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
    def __init__(self,alt, errRate,phis=PHIS,theta=THETA,f=FOCAL_LENGTH, imX=24., imY=36.0, pixX=20, pixY=30, rnge=300.):
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