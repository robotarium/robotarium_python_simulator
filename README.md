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

The guide below will show you how to install the necessary dependencies. The simulator has been thoroughly tested on Python 3.5.x+ versions.

### Linux
To install the simulator on linux requires the installation of the dependencies labeled above. The installation varies depending on the distribution used. The easiest way to install CVXOPT is to use pip, which is typically installed with the default python installation.

#### Ubuntu, Debian, and other Ubuntu/Debian based distributions.
```
# Automatically
Navigate to the cloned simulator directory containing the setup.py script. Then run,
pip3 install .
**Note the dot after install** 

# Manually
sudo apt-get install python3-numpy python3-scipy python3-matplotlib python3-pip
pip3 install cvxopt --user
```

#### Fedora, CentOS, and other RPM based distributions.
```
# Python 3.5.x+
sudo yum install numpy scipy python3-matplotlib python3-pip  # For YUM package manager.
sudo dnf install numpy scipy python3-matplotlib python3-pip  # For DNF package manager.
pip3 install cvxopt --user
```

#### pip
If you are already using python with (or without) pip installed and configured, the installation can be done simply with the following commands:

```
# Python 3.5.x+
sudo apt-get install python3-pip  # Ubuntu/Debian based
sudo yum install python3-pip  # Fedora/CentOS based (RPM Yum based)
sudo dnf install python3-pip  # Fedora/CentOS based (RPM dnf based)

pip3 install scipy
pip3 install numpy
pip3 install matplotlib
pip3 install cvxopt --user
```
### Windows
Of the three installations, this one will be the most difficult due to the fact that Windows does not come with a native or easily installable package manager. To circumvent these problems, it will be necessary to install the packages using pip. The issue with using pip, however, is that NumPy, SciPy, and matplotlib require the packages to be installed without compiling. Therefore, each wheel must be installed individually. This is a simple process using pip 8.x version. The following commands are for python installations that are using PIP 8.x version. The wheel files used here can be found [here] (http://www.lfd.uci.edu/~gohlke/pythonlibs/).

NOTE: The following files installed are for 64-bit architectures. If you have a 32-bit CPU, download the corresponding 32-bit and python versions of the files specified below.

#### Install NumPy
It is important to note the naming conventions and install the correct version according to your python version.
```
# Install NumPy (64-bit)
pip install numpy-1.11.1+mkl-cp27-cp27m-win_amd64.whl  # Python 2.7.x Version
pip install numpy-1.11.1+mkl-cp34-cp34m-win_amd64.whl  # Python 3.4.x Version
pip install numpy-1.11.1+mkl-cp35-cp35m-win_amd64.whl  # Python 3.5.x Version

# Install NumPy (32-bit)
pip install numpy-1.11.1+mkl-cp27-cp27m-win32.whl  # Python 2.7.x Version
pip install numpy-1.11.1+mkl-cp34-cp34m-win32.whl  # Python 3.4.x Version
pip install numpy-1.11.1+mkl-cp35-cp35m-win32.whl  # Python 3.5.x Version
```

#### Install SciPy
It is important to note the naming conventions and install the correct version according to your python version.
```
# Install SciPy (64-bit)
pip install scipy-0.18.0-cp27-cp27m-win_amd64.whl  # Python 2.7.x Version
pip install scipy-0.18.0-cp34-cp34m-win_amd64.whl  # Python 3.4.x Version
pip install scipy-0.18.0-cp35-cp35m-win_amd64.whl  # Python 3.5.x Version

# Install SciPy (32-bit)
pip install scipy-0.18.0-cp27-cp27m-win32.whl  # Python 2.7.x Version
pip install scipy-0.18.0-cp34-cp34m-win32.whl  # Python 3.4.x Version
pip install scipy-0.18.0-cp35-cp35m-win32.whl  # Python 3.5.x Version
```

#### Install matplotlib
Installation of matplotlib requires extra dependencies to be installed first. Again, it is important to note the naming conventions for the matplotlib module and install the correct version according to your python version.

```
# Install dateutil
pip install python_dateutil-2.5.3-py2.py3-none-any.whl

# Install pytz
pip install pytz-2016.6.1-py2.py3-none-any.whl

# Install pyparsing
pip install pyparsing-2.1.8-py2.py3-none-any.whl

# Install cycler
pip install cycler-0.10.0-py2.py3-none-any.whl

# Install setuptools
pip install setuptools-25.2.0-py2.py3-none-any.whl

# Install matplotlib (64-bit)
pip install matplotlib-1.5.2-cp27-cp27m-win_amd64.whl  # Python 2.7.x Version 
pip install matplotlib-1.5.2-cp34-cp34m-win_amd64.whl  # Python 3.4.x Version
pip install matplotlib-1.5.2-cp35-cp35m-win_amd64.whl  # Python 3.5.x Version

# Install matplotlib (32-bit)
pip install matplotlib-1.5.2-cp27-cp27m-win32.whl  # Python 2.7.x Version 
pip install matplotlib-1.5.2-cp34-cp34m-win32.whl  # Python 3.4.x Version
pip install matplotlib-1.5.2-cp35-cp35m-win32.whl  # Python 3.5.x Version
```

#### Install the Robotarium Module
# Install RPS
In your command terminal, navigate to the cloned python simulator repository containing setup.py. Then run,

pip install .

### macOS
To install the simulator on macOS, it is recommended to install a package manager for easy installation. CVXOPT will have to be installed using PIP.

#### Homebrew
To use [Homebrew] (http://brew.sh) for dependency installation requires a bit of extra work due to the scipy stack not being a part of the main repository. You can then install the dependencies labeled above using the following work around (Requires PIP). A more detailed explanation can be found [here] (https://penandpants.com/2012/02/24/install-python/).

```
# Install Python (Choose Python 2.7.x or 3.5.x)
brew install python
brew install python3

# Restart terminal to allow the path to python to be updated.
# make sure "which python" command returns "/usr/local/bin/python"

# Install pip
easy_install pip

# Install NumPy
pip install numpy

# Install SciPy
brew install gfortran  # Install to prevent an error inherent in SciPy.
pip install scipy

# Install matplotlib
brew install pkg-config
pip install matplotlib

# Install CVXOPT
pip install cvxopt --user
```

#### Macports
To use [Macports] (https://www.macports.org/), use the following commands to install the scipy stack. At the time of writing, a Python 3.5.x version for the NumPy stack do not exist. 
```
# For Python 2.7+
sudo port install py27-numpy py27-scipy py27-matplotlib

# Install pip
easy_install pip

# Install CVXOPT
pip install cvxopt --user
```

## Usage
To run one of the examples:

 ```
 python "path_to_simulator"/rps/examples/plotting/barrier_certificates_with_plotting.py
 ```

## Issues
Please enter a ticket in the [issue tracker](https://github.com/robotarium/robotarium_python_simulator/issues).

