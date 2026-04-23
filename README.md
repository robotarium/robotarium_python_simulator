# Robotarium Python Simulator
 
A Python simulator and API for usage of the [Robotarium](https://www.robotarium.gatech.edu/) — a remotely accessible multi-robot testbed at Georgia Institute of Technology. This is an open-source re-implementation of the [MATLAB simulator](https://github.com/robotarium/robotarium-matlab-simulator).

Introduction
============

This is a Python simulator for Robotarium! The Robotarium is a project at Georgia Institute of Technology allowing public, remote access to a state-of-the-art multi-robot testbed.

## Requirements
 
- Python 3.10+
- NumPy
- Matplotlib
- CVXOPT

## Installation
The simulator can run on all major platforms (Windows, Linux, and macOS). All that is required is cloning the repository and installing some necessary dependencies.

```bash
git clone https://github.com/robotarium/robotarium_python_simulator
cd robotarium_python_simulator
pip install .
```
 
Or install dependencies manually then install the package:
 
```bash
pip install numpy matplotlib cvxopt
pip install .
```
**Note the dot after install**

## Dependency Installation

The guide below will show you how to install the necessary dependencies. The simulator has been thoroughly tested on Python 3.10.x+ versions.

### Submission Dependencies

The current list of libraries supported by the robotarium is contained in [libraries.txt](./libraries.txt).  If you cannot find a library that is required for your submission, please submit a pull request adding your package to the libraries.txt file so the team can evaluate its addition into the robotarium.


## Usage
To run one of the examples:

 ```
 python "path_to_simulator"/rps/examples/plotting/barrier_certificates_with_plotting.py
 ```

## Package Structure
 
```
rps/
├── robotarium.py              # Main simulator class
├── robotarium_abc.py          # Abstract base class
├── utilities/
│   ├── barrier_certificates.py  # SI and unicycle barrier certificates
│   ├── controllers.py           # SI and unicycle position/pose controllers
│   ├── transformations.py       # SI <-> unicycle mappings
│   ├── graph.py                 # Graph Laplacian utilities
│   ├── misc.py                  # Pose generation and convergence checkers
│   └── sensors.py               # Distance sensor simulation
├── patch_creation/
│   └── gternal_patch.py         # GTernal robot visualization
└── examples/
    ├── si_barriers.py
    ├── uni_barriers.py
    ├── sensor_readings.py
    ├── dead_reckoning.py
    ├── plotting/
    │   ├── barriers_with_plotting.py
    │   ├── go_to_pose_with_plotting.py
    │   └── leader_follower_with_plotting.py
```

## Issues
Please enter a ticket in the [issue tracker](https://github.com/robotarium/robotarium_python_simulator/issues).

