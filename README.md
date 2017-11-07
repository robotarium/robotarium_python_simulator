# robotarium_python_simulator
A Python simulator for the Robotarium!

# Alpha Warning 
Warning: This simulator is currently in "alpha", so there may be some restructuring, renaming, and general changes.  For best results, **clone frequently.**  

There shouldn't be any ground-breaking changes, but some restructuring may occur as we work on the backend code that's associated with the simulator.

# Installation (Linux)
This simulator is built using Python 3.5, so you'll need to install that on your system.

We haven't been able to test with previous versions yet, but anything 3.x should work. 

To install, navigate the directory in which you cloned the simulator.  Then, run 
```
sudo pip3 install .
```
This instruction will install the necessary packages and the simulator on your system.  

# Running without installation 
You can also just install the required modules (numpy et al.) and run without install this package.  Simply navigate to the robotairum_python_simulator directory and run 
```
python3 rps/examples/consensus.py
```

# Dependencies
The dependencies are:
* scipy
* numpy
* cvxopt
* matplotlib
Though, these are included in the setup.py file.  For the absolutely correct list, **see the setup.py file.**

# Examples
There are a variety of examples in the 'examples' folder.  Check these out before
starting your simulation!

# Submission Instructions
When your simulation is complete, go the the Robotarium [website!][https://www.robotarium.org], follow the instructions there, and see your code run on real robots!
