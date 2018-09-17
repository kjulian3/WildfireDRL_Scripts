from QNetwork import *
from Simulation import *
from ReplayMemory import *
    
## Set Parameters
TRAIN_FREQ  = 10   # Number of samples to generate between trainings (Should be multiple of 10)
PRINT_FREQ  = 100  # Frequency of printing (Should be a multiple of 10)
DT          = 0.5  # Time between wildfire updates            
DTI         = 0.1  # Time between aircraft decisions

    
''' Train Network '''
repMem = ReplayMemory()        # Object representing replay memory
q = QNetwork(repMem.init_size) # Object representing the Q-Network we are training. Contains a model and target network
sim = Sim(DT,DTI)              # Simulation environment, which contains wildfire, aircraft, and observation filter models

# Loop forever 
count=0
while True:
    
    # Number of loops before updating network. Two samples are generated every loop
    for j in range(TRAIN_FREQ/int(2*DT/DTI)):
        
        sim.stepFires()                                                           # Advance wildfires by one step
        state0 = sim.getBeliefAndState(0,1); state1 = sim.getBeliefAndState(1,0); # Update relative wildfire states
        
        #Update aircraft and store state-action-reward-next state tuples in replay memory
        for i in range(int(DT/DTI)):
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
            
    # Printing
    if (count % PRINT_FREQ) ==0 and count>=repMem.init_size:
        print "Samples: %d, Trainings: %d" % (count,(count-repMem.init_size)/TRAIN_FREQ),"Loss: %.3e" % q.test(repMem.sample_batch(q.batch_size))
    elif (count % 100) ==0 and count<repMem.init_size:
        print "Samples: %d" % count
        
    # Train network on a batch from replay memory
    if count>=repMem.init_size:
        q.train(repMem.sample_batch(q.batch_size))