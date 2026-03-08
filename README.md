Introduction
============

This is a Python simulator for Robotarium! The Robotarium is a project at Georgia Institute of Technology allowing public, remote access to a state-of-the-art multi-robot testbed.

This is an open-source, re-implementation of the [MATLAB simulator]  (https://github.com/robotarium/robotarium-matlab-simulator) in Python.

## Installation
The simulator can run on all major platforms (Windows, Linux, and macOS). All that is required is cloning the repository and installing some necessary dependencies.

```
https://github.com/robotarium/robotarium_python_simulator
```

The following dependencies are required for utilization of the simulator:
- [NumPy] (http://www.numpy.org)
- [matplotlib] (http://matplotlib.org/index.html)
- [CVXOPT] (http://cvxopt.org/index.html)

NOTE: The SciPy stack and matplotlib can be difficult to install on Windows. However, [this] (http://www.lfd.uci.edu/~gohlke/pythonlibs/) link provides .whl files necessary for installation. Make sure to install all the dependencies for each version part of the SciPy and matplotlib stack!

## Dependency Installation

The guide below will show you how to install the necessary dependencies. The simulator has been thoroughly tested on Python 3.10.x+ versions.


### Pip

Pip is the standard dependency manager for python.  To install the simulator, use
```
# Install Dependencies
pip install numpy==2.2.6 matplotlib==3.10.8 cvxopt==1.3.2

# Installing the Robotarium Simulator
# Navigate to the cloned simulator directory containing the setup.py script. Then run:
pip install .
**Note the dot after install**
```

### Submission Dependencies

The current list of libraries supported by the robotarium is contained in [libraries.txt](./libraries.txt).  If you cannot find a library that is required for your submission, please submit a pull request adding your package to the libraries.txt file so the team can evaluate its addition into the robotarium.


## Usage
To run one of the examples:

 ```
 python "path_to_simulator"/rps/examples/plotting/barrier_certificates_with_plotting.py
 ```

## Issues
Please enter a ticket in the [issue tracker](https://github.com/robotarium/robotarium_python_simulator/issues).

