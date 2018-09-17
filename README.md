# WildfireDRL_Scripts
Scripts for training and evaluating neural networks used in the Wildfire DRL papers

### Overview
The code is divided into separate files:
* **TrainNetwork.py**: Trains the neural network controller
* **ReplayMemory.py**: Represents the Q-learning replay memory
* **QNetwork.py**: Contains all of the neural network code (Keras running on Theano), including an object for training Q-Networks
and an object to read and evaluated trained network HDF5 files from Keras

* **Simulation.py**: Represents the simulation environment
* **AircraftModel.py**: Represents the aircraft state, dynamics, and observation model
* **Filters.py**: Describes the EKF and particle filter approachs for filtering noisy camera observations
* **WildfireModels.py**: Contains methods for a stochastic wildfire model

### Prerequisites
These files use python with the following dependencies:
* **numpy**
* **math**
* **Theano**
* **Keras** (using the Theano backend)
