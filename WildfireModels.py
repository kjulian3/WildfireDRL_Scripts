import numpy as np

AVG_FUEL=18.0
BURN_THRESH = 22.0
INCLUDE_ARC = False

'''
Wildfire model
'''
class FireModel(object):
    def __init__(self,avgFuel=AVG_FUEL,bt=BURN_THRESH,includeArc=INCLUDE_ARC,seed=None,windx=-2.0,windy=-2.0,burnMap=None,fuelMap=None):
        
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
    def __init__(self,windx,windy,avgFuel=AVG_FUEL,bt=BURN_THRESH,includeArc=INCLUDE_ARC,burnMapProbs=None,fuelMapProbsList=None):
            
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