from cvxopt import matrix, solvers
import quadprog as solver2
import numpy as np
import time
solvers.options['show_progress'] = False
solvers.options['reltol'] = 1e-2 # was e-2
solvers.options['feastol'] = 1e-2 # was e-4
solvers.options['maxiters'] = 50 # default is 100

def create_robust_barriers(max_num_obstacles = 100, max_num_robots = 30, d = 5, wheel_vel_limit = 12.5, base_length = 0.105, wheel_radius = 0.016,
    projection_distance =0.05, gamma = 150, safety_radius = 0.12): # gamma was 150
    D = np.matrix([[wheel_radius/2, wheel_radius/2], [-wheel_radius/base_length, wheel_radius/base_length]])
    L = np.matrix([[1,0],[0,projection_distance]])* D
    disturb = np.matrix([[-d, -d, d, d],[-d, d, d, -d]])
    num_disturbs = np.size(disturb[1,:])

    #TODO: Take out 4*max_num_robots?
    max_num_constraints = (max_num_robots**2-max_num_robots)//2 + max_num_robots*max_num_obstacles + 4*max_num_robots
    A = np.matrix(np.zeros([max_num_constraints, 2*max_num_robots]))
    b = np.matrix(np.zeros([max_num_constraints, 1]))
    Os = np.matrix(np.zeros([2,max_num_robots]))
    ps = np.matrix(np.zeros([2,max_num_robots]))
    Ms = np.matrix(np.zeros([2,2*max_num_robots]))

    def robust_barriers(dxu, x, obstacles):

        num_robots = np.size(dxu[0,:])

        if obstacles.size != 0:
            num_obstacles = np.size(obstacles[0,:])
        else:
            num_obstacles = 0

        if(num_robots < 2):
            temp = 0
        else:
            temp = (num_robots**2-num_robots)//2


        if num_robots == 0:
            return []


        # Generate constraints for barrier certificates based on the size of the safety radius
        num_constraints = temp + num_robots*num_obstacles
        A[0:num_constraints, 0:2*num_robots] = 0
        Os[0, 0:num_robots] = np.cos(x[2, :])
        Os[1, 0:num_robots] = np.sin(x[2, :])
        ps[:, 0:num_robots] = x[0:2, :] + projection_distance*Os[:, 0:num_robots]
        # Ms Real Form
        # Ms[0, 0:2*num_robots:2] = Os[0, 0:num_robots]
        # Ms[0, 1:2*num_robots:2] = -projection_distance*Os[1, 0:num_robots]
        # Ms[1, 1:2*num_robots:2] = projection_distance*Os[0, 0:num_robots]
        # Ms[1, 0:2*num_robots:2] = Os[1, 0:num_robots]
        # Flipped Ms to be able to perform desired matrix multiplication
        Ms[0, 0:2*num_robots:2] = Os[0, 0:num_robots]
        Ms[0, 1:2*num_robots:2] = Os[1, 0:num_robots]
        Ms[1, 1:2*num_robots:2] = projection_distance*Os[0, 0:num_robots]
        Ms[1, 0:2*num_robots:2] = -projection_distance*Os[1, 0:num_robots]
        MDs  = (Ms.T * D).T
        temp = np.copy(MDs[1, 0:2*num_robots:2])
        MDs[1, 0:2*num_robots:2] =  MDs[0, 1:2*num_robots:2]
        MDs[0, 1:2*num_robots:2] = temp

        count = 0

        for i in range(num_robots-1):
            diffs = ps[:,i] - ps[:, i+1:num_robots]
            hs = np.sum(np.square(diffs),0) - safety_radius**2 # 1 by N
            h_dot_is = 2*diffs.T*MDs[:,2*i:2*i+2] # N by 2
            h_dot_js = np.matrix(np.zeros((2,num_robots - (i+1))))
            h_dot_js[0, :] = -np.sum(2*np.multiply(diffs, MDs[:,2*(i+1):2*num_robots:2]), 0)
            h_dot_js[1, :] = -np.sum(2*np.multiply(diffs, MDs[:,2*(i+1)+1:2*num_robots:2]), 0)
            new_constraints = num_robots - i - 1
            A[count:count+new_constraints, (2*i):(2*i+2)] = h_dot_is
            A[range(count,count+new_constraints), range(2*(i+1),2*num_robots,2)] = h_dot_js[0,:]
            A[range(count,count+new_constraints), range(2*(i+1)+1,2*num_robots,2)] = h_dot_js[1,:]
            b[count:count+new_constraints] = -gamma*(np.power(hs,3)).T - np.min(h_dot_is*disturb,1) - np.min(h_dot_js.T*disturb,1)
            count += new_constraints

        if obstacles.size != 0:
            # Do obstacles
            for i in range(num_robots):
                diffs = (ps[:, i] - obstacles)
                h = np.sum(np.square(diffs),0) - safety_radius**2
                h_dot_i = 2*diffs.T*MDs[:,2*i:2*i+2]
                A[count:count+num_obstacles,(2*i):(2*i+2)] = h_dot_i
                b[count:count+num_obstacles] = -gamma*(np.power(h,3)).T  - np.min(h_dot_i*disturb, 1)
                count = count + num_obstacles

        # Adding Upper Bounds On Wheels
        A[count:count+2*num_robots,0:2*num_robots] = -np.eye(2*num_robots)
        b[count:count+2*num_robots] = -wheel_vel_limit
        count += 2*num_robots
        # # Adding Lower Bounds on Wheels
        A[count:count+2*num_robots,0:2*num_robots] = np.eye(2*num_robots)
        b[count:count+2*num_robots] = -wheel_vel_limit
        count += 2*num_robots

        # Solve QP program generated earlier
        L_all = np.kron(np.eye(num_robots), L)
        dxu = np.linalg.inv(D)*dxu # Convert user input to differential drive
        vhat = np.matrix(np.reshape(dxu ,(2*num_robots,1), order='F'))
        H = 2*L_all.T*L_all
        f = np.transpose(-2*np.transpose(vhat)*np.transpose(L_all)*L_all)

        # Alternative Solver
        #start = time.time()
        #vnew2 = solvers.qp(matrix(H), matrix(f), -matrix(A[0:count,0:2*num_robots]), -matrix( b[0:count]))['x'] # , A, b) Omit last 2 arguments since our QP has no equality constraints
        #print("Time Taken by cvxOpt: {} s".format(time.time() - start))

        start = time.time()
        vnew = solver2.solve_qp(H, -np.squeeze(np.array(f)), A[0:count,0:2*num_robots].T, np.squeeze(np.array(b[0:count])))[0]
        #print("Time Taken by quadprog: {} s".format(time.time() - start))
        # Initial Guess for Solver at the Next Iteration
        # vnew = quadprog(H, double(f), -A(1:num_constraints,1:2*num_robots), -b(1:num_constraints), [], [], -wheel_vel_limit*ones(2*num_robots,1), wheel_vel_limit*ones(2*num_robots,1), [], opts);
        # Set robot velocities to new velocities
        dxu = np.reshape(vnew, (2, num_robots), order='F')
        dxu = D*dxu

        return dxu

    return robust_barriers
