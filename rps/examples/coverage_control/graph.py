import matplotlib.pyplot as plt
import pandas as pd
file1 = pd.read_csv('cost_case1.csv')
file2 = pd.read_csv('cost_case4.csv')
file3 = pd.read_csv('cost_case5.csv')
file4 = pd.read_csv('cost_case3.csv')

plt.figure(figsize=(12,8))
plt.plot(file1, label='Locational Cost', color='b')
plt.plot(file2, label='Temporal Cost', color='g')
plt.plot(file3, label='Power Cost', color='r')
plt.plot(file4,label='Range Limited Cost', color='y')

# Adding labels and title
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Line Plot for Cost Cases 1, 2,3 and 4')
plt.legend()
plt.grid(True)

plt.show()