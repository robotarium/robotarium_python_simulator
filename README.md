# robotarium_python_simulator
A Python simulator for the Robotarium!

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
