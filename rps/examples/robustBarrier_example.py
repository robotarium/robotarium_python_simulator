from rps.utilities.barrier_certificates2 import *
import numpy as np

robust_barriers = create_robust_barriers()
x = np.array([[0.0, 0.15],[0.0, 0.0],[np.pi/6, -5*np.pi/6]])
dxu = np.array([[0.2, 0.2],[0.0, 0.0]])
dxu_adjusted = robust_barriers(dxu, x, np.array([]))
print("x = ")
print(x)
print("dxu = ")
print(dxu)
print("dxu_adjusted = ")
print(dxu_adjusted)

# Output Matches MATLAB: TRUE
# Output of MATLAB's barrier_certificates2
# dxu_adjusted =
#
#     0.0844    0.0844
#     2.2018    2.2018

# Output of Python robust_barriers
# dxu_adjusted =
# [[0.08440613 0.08440613]
#  [2.20178797 2.20178797]]
