import numpy as np

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
