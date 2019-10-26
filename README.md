Introduction
============

This is a Python simulator for Robotarium! The Robotarium is a project at Georgia Tech allowing public, remote access to a state-of-the-art multi-robot testbed.

This is an open-source, re-implementation of the [MATLAB simulator]  (https://github.com/robotarium/robotarium-matlab-simulator) in Python. The purpose of this project is to allow people to further experiment with the Robotarium simulator in the following ways:

1. Unable to have access to MATLAB.
2. Want the flexibility of Python.
3. Educational purposes, such as learning about multi-agent systems.

## Installation
The simulator should be able to run on all major platforms (Windows, Linux, and macOS). However, it has only been thoroughly tested on Linux with Python3.

```
git clone https://github.com/zmk5/robotarium_python_simulator.git
```

The following dependencies are required for utilization of the simulator:
- [NumPy] (http://www.numpy.org)
- [matplotlib] (http://matplotlib.org/index.html)
- [CVXOPT] (http://cvxopt.org/index.html)
- [quadprog] (https://pypi.org/project/quadprog/)

NOTE: The SciPy stack and matplotlib can be difficult to install on Windows. However, [this] (http://www.lfd.uci.edu/~gohlke/pythonlibs/) link provides .whl files necessary for installation. Make sure to install all the dependencies for each version part of the SciPy and matplotlib stack!

## Linux

The linux instalation should be fairly easy. Open a terminal, navigate to the folder the repository was cloned into. And run,

'''
pip3 install .
'''

