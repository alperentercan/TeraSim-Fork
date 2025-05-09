'''


TODO:
1. nominal acc controller can be better like lqr.
'''


import numpy as np
from scipy.io import loadmat
from pathlib import Path
import cvxpy as cp
from controllers.control_utils import two_point_distance, wrap_angle
from controllers.cav_utils import *
import math 

CWD = str(Path(__file__).parent)
print(CWD)

class AlphaFilter:
    def __init__(self, H_max, h_max, H_xu, h_xu):
        self.f = True
        self.H_max = H_max
        self.h_max = h_max
        self.H_xu = H_xu
        self.h_xu = h_xu
        self.u_pre = [0, 0]  # previous input

        self.pos_ind = H_max[:, -1] > 0
        self.zero_ind = H_max[:, -1] == 0
        self.neg_ind = H_max[:, -1] < 0
        if not np.any(self.pos_ind):
            self.pos_ind = None
        if not np.any(self.neg_ind):
            self.neg_ind = None

    def get_max_alpha(self, x):
        '''
        Compute the maximal alpha given the current state

        Inputs: x (np.Array)--- the state of one subsystem, 3x1
                H_max, h_max (np.Array) --- [H_max, h_max] is the H-representation 
                                of the maximal RCIS for the alpha dynmaics
        Return: alpha --- the maximal alpha at x
        '''
        A = self.H_max[:, -1]
        b = self.h_max - self.H_max[:, 0:-1] @ x
        alpha_max = np.inf
        alpha_min = -np.inf
        if self.pos_ind is not None:
            alpha_max = np.min(b[self.pos_ind].squeeze() / A[self.pos_ind])
        if self.neg_ind is not None:
            alpha_min = np.max(b[self.neg_ind].squeeze() / A[self.neg_ind])

        if (alpha_max < alpha_min) or (np.any(b[self.zero_ind] < -1e-6)):
            alpha_max = -np.inf
            alpha_min = -np.inf

        return alpha_max, alpha_min

    def qp(self, x, u_ref):
        '''
        Solve the QP program in one subsystem.
        Inputs: x (np.array) ---  the states of one subsystem (i.e. x, vx, alpha)
                u_ref (float) --- the reference input
        Return: u_corr (float)--- the supervised input
        '''
        u = cp.Variable(2)

        H = self.H_xu[:, 3:]
        h = self.h_xu - self.H_xu[:, 0:3] @ x
        prob = cp.Problem(cp.Minimize((u[0] - u_ref) ** 2),
                          [H @ u <= h.squeeze()])
        prob.solve()
        # prob.solve(cp.PROXQP)
        try:
            return u.value[0]
        except TypeError:
            print("QP solver error!", x)
            return u_ref


    def eval(self, x, u_ref):
        '''
        Project the reference input to the admissible input set at x.
        Inputs: xy (list) --- the states of the 4d system (x, y, vx, vy)
                u_ref (list) --- the reference input, 1x2
        Return: u (list) --- the supervised input, 1x2
        '''
        alpha_x, _ = self.get_max_alpha(x)

        if alpha_x == -np.inf:
            ux = u_ref
        else:
            ux = self.qp(np.vstack((x, alpha_x)), u_ref)

        self.u_pre = ux 
        return ux



class Controller:
    def __init__(self, waypoints:np.ndarray):
        '''
        Initialize the controller.
        '''
        # K matrix for steering control
        self.K = np.array([0.2128, -0.0015, 0.9014, -0.0511])

        # minimal headway
        self.h_min = 15 

        # input limits
        self.acc_max = 2.87 
        self.acc_min = -7.06

        self.steering_min = -0.4*np.pi
        self.steering_max = 0.4*np.pi

        # target velocity
        self.v_target = 15 

        # waypoints
        self.waypts = waypoints 
        self.waypt_id = 1 # the initial waypoint 
        
        # alpha filter
        data = loadmat(CWD + '/data/alpha_filter_py.mat')
        self.alpha_filter = AlphaFilter(data['H_max'], data['h_max'], data['H_xu'], data['h_xu'])

    def steering(self, states: np.array):
        '''
        Input: states = [lateral displacement, lateral velocity, yaw, yaw rate]
        Output: steering angle
        ''' 
        # nominal steering value
        steer = np.dot(states, self.K)
        return max(min(steer, self.steering_max), self.steering_min)
    
    def throttle(self, states: np.array):
        '''
        Input: states = [headway, ego longitudinal velocity, lead velocity] 
        Output: longitudinal acceleration
        '''
        headway, v_ego, v_lead = states
        if headway < self.h_min:
            # if headway is less than acceptable minimum headway
            if v_ego > 0:
                # if ego is moving forward, emergency break
                return self.acc_min
            else:
                # if ego is stopping, stay there
                return 0
        else:
            # nominal acceleration to achieve target speed
            acc = -0.3*(v_ego - self.v_target)
            
            if headway != float('inf'):
                # if there is a lead vehicle, compute the corrected input 
                # print("alpha filter activated!")
                delta_v = v_lead - v_ego
                acc_new = self.alpha_filter.eval(np.array([[headway], [delta_v]]), acc)
            else:
                # if no lead vehicle, nominal input is usable
                acc_new = acc
            
            return max(self.acc_min, min(self.acc_max, acc_new))
    
    def compute_control(self, ego_state:State, bv_states:dict):
        '''
        
        ''' 
        # get the next waypoint
        print(f"Ego state {ego_state.x,ego_state.y}")
        self.update_waypoint_id(ego_state.x, ego_state.y)
        
        if self.waypt_id == (len(self.waypts)-1):
            # if reach the end of the reference, stop immediately
            if ego_state.v_long > 0:
                return (self.acc_min, 0)
            else:
                return (0, 0)
            
        # acceleration input
        headway, v_lead = self.get_headway(ego_state, bv_states)
        acc_states = np.array([headway, ego_state.v_long, v_lead])
        acc = self.throttle(acc_states)

        # steering input
        # RK: this part is directly from Zexiang's Controller, 
        # I didn't check if it makes sense.
        lookahead_id = min(self.waypt_id + 2, len(self.waypts)-1) 

        current_wp_x = self.waypts[self.waypt_id][0]
        current_wp_y = self.waypts[self.waypt_id][1]
        
        next_wp_x = self.waypts[lookahead_id][0]
        next_wp_y = self.waypts[lookahead_id][1]

        steer_adjustment = (
            math.atan2(next_wp_y-current_wp_y, 
                    next_wp_x-current_wp_x)
            -np.pi/2
            )

        y = (
            (ego_state.x-current_wp_x)*np.cos(steer_adjustment) 
            + (ego_state.y-current_wp_y)*np.sin(steer_adjustment))
        
        v_lat_adjusted = (
            (np.cos(steer_adjustment)
            *(ego_state.v_long*np.cos(ego_state.theta)
                + ego_state.v_lat*np.sin(ego_state.theta)) )
            +(np.sin(steer_adjustment)
                *(ego_state.v_long*np.sin(ego_state.theta)
                    -ego_state.v_lat*np.cos(ego_state.theta)))
        )
        
        theta_adjusted = (steer_adjustment + np.pi/2) - ego_state.theta
        theta_adjusted = wrap_angle(theta_adjusted)

        steer_states = np.array([y, v_lat_adjusted ,theta_adjusted, -ego_state.theta_dot])
        steering = self.steering(steer_states)
        return (acc, steering)
        
    def get_waypoint_id(self):
        '''
        Get the waypoint id
        '''
        return self.waypt_id

    def update_waypoint_id(self, x, y):
        '''
        Find the nearest waypoint, given the position of the ego
        '''
        if self.waypt_id == -1:
            # initialize waypt_id: search the nearest waypoint globally
            min_dist = float('inf')
            for i in range(len(self.waypts)):
                dist = two_point_distance(x, y, self.waypts[i][0], self.waypts[i][1])
                if dist < min_dist:
                    self.waypt_id = i
                    min_dist = dist
        else:
            # update the nearest waypoint locally
            min_dist =  two_point_distance(x, y, self.waypts[self.waypt_id][0], self.waypts[self.waypt_id][1])
            new_id = 0 #self.waypt_id + 1
            if new_id >= len(self.waypts):
                return
            new_dist = two_point_distance(x, y, self.waypts[new_id][0], self.waypts[new_id][1])
            while(new_dist <= min_dist):
                # only search waypoints ahead of the current pos
                self.waypt_id = new_id
                min_dist = new_dist
                new_id = new_id + 1
                if new_id >= len(self.waypts):
                    return
                new_dist = two_point_distance(x, y, self.waypts[new_id][0], self.waypts[new_id][1])

    def get_headway(self, ego, bvs):
        '''
        Find the headway distance. 
        '''
        min_dist = 10**4
        headway = np.inf
        v_lead = np.inf
        for bv in bvs:
            if np.abs(ego.theta - bv.theta) <= np.pi/10:
                # if bv and ego go approximately in the same orientation
                dist = two_point_distance(ego.x, ego.y, bv.x, bv.y)
                if dist < 100:
                    # if bv is in 100m distance, check if bv is ahead of ego
                    position = np.array([ego.x, ego.y]) + dist*np.array([np.cos(ego.theta), np.sin(ego.theta)])
                    # print(dist, position, ego.x, ego.y, bv.x, bv.y)
                    if two_point_distance(position[0], position[1], bv.x, bv.y) <= 10:
                        # if bv is ahead of ego
                        if dist < min_dist:
                            # print("min dist changed!")
                            min_dist = dist
                            headway = dist
                            v_lead = bv.v_long

        return headway, v_lead

        
