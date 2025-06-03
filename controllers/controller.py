'''


TODO:
1. nominal acc controller can be better like lqr.
'''


import numpy as np
from scipy.io import loadmat
from pathlib import Path

from controllers.src.control_utils import two_point_distance, wrap_angle
from controllers.src.cav_utils import *
from controllers.src.lateral_control import *
from controllers.src.longitudinal_control import *
import math 

CWD = str(Path(__file__).parent)
print(CWD)


class Controller:
    def __init__(self, waypoints:np.ndarray, dt=0.1):
        '''
        Initialize the controller.
        '''
        # load the limits and targets
        # TODO: 
        # have a limits config file and read the values from there
        # minimal headway
        self.h_min = 15 

        # input limits
        self.acc_max = 2.87 
        self.acc_min = -7.06

        self.steering_min = -2.5*np.pi
        self.steering_max = 2.5*np.pi
        self.steering_ratio = 16

        # target velocity
        self.v_target = 20 


        self.dt = dt, 
        # load waypoints
        self.waypts = waypoints 
        self.waypt_id = 1 # the initial waypoint 
        
        # add alpha filter for longitudinal control
        data = loadmat(CWD + '/data/alpha_filter_py.mat')
        self.alpha_filter = AlphaFilter(data['H_max'], data['h_max'], data['H_xu'], data['h_xu'])

        # initialize the lateral controller
        self.lateral_controller = PreviewLQR()

    def steering(self, states: np.ndarray, control_states):
        '''
        Lateral controller to compute the steering angle.

        Input: 
        - states (np.ndarray) : States for the controller [lateral displacement, lateral velocity, yaw, yaw rate]
        
        Output: 
        - steering (float): Steering angle in radians
        ''' 
        assert isinstance(states, State), "States should be an instance of State"
        assert isinstance(control_states, StatesLat), "Control states should be an instance of StatesLat"

        # nominal steering value
        steering = self.lateral_controller.run(states, control_states)
        steering *= self.steering_ratio  
        if steering < self.steering_min:
            print("Steering value is below the minimum limit, setting to minimum.")
            return self.steering_min
        elif steering > self.steering_max:
            print("Steering value is above the maximum limit, setting to maximum.")
            return self.steering_max
        else:
            return steering

    def acceleration(self, states: StatesLong):
        '''
        Longitudinal controller to compute the acceleration input.
        
        Input:
        - states (np.ndarray) = States for the controller [headway, ego longitudinal velocity, lead velocity] 
        
        Output:
        - longitudinal acceleration (float): Longitudinal acceleration in m/s^2
        '''
        # TODO : have an LQR maybe for the nominal acceleration
        assert isinstance(states, StatesLong), "States should be an instance of StatesLong"

        # check if the headway is less than the minimum acceptable headway
        if states.hw < self.h_min:
            # headway is less than acceptable min headway -> emergency break
            if states.v_long > 0:
                return self.acc_min # ego is moving, apply max break
            else:
                return 0  # ego has stopped, no need to break
        else:
            acc = -0.5*(states.v_long - self.v_target) # nominal acceleration to achieve target speed
            
            # if there is a lead vehicle, compute the corrected input 
            if states.hw != float('inf'):
                dhw = states.v_lead - states.v_long # speed difference, in other words, headway rate
                alpha_filter_states = np.array([[states.hw], [dhw]])
                acc = self.alpha_filter.eval(alpha_filter_states, acc)
            
            if acc > self.acc_max:
                print("Acceleration exceeds maximum limit, setting to maximum.")
                return self.acc_max
            elif acc < self.acc_min:
                print("Acceleration is below minimum limit, setting to minimum.")
                return self.acc_min
            else:
                return acc

    
    def compute_control(self, ego_state:State, bv_states:dict, previous_lat_control:StatesLat=None):
        '''
        Compute the control inputs for the ego vehicle.

        Inputs:
        - ego_state (State): The state of the ego vehicle
        - bv_states (dict): A dictionary of states of the surrounding vehicles, where keys are vehicle IDs and values are State objects
        
        Outputs:
        - (acceleration, steering) : A tuple containing the acceleration input and steering angle
        ''' 

        current_pos = [ego_state.x, ego_state.y]
        # get the next waypoint
        self.update_waypoint_id(ego_state.x, ego_state.y)
        
        if self.waypt_id == (len(self.waypts)-1):
            # if reach the end of the reference, stop immediately
            if ego_state.v_long > 0:
                return (self.acc_min, 0)
            else:
                return (0, 0)
        
        # acceleration input
        headway, v_lead = self.get_headway(ego_state, bv_states)
        acc_states = StatesLong(headway, ego_state.v_long, v_lead)
        acc = self.acceleration(acc_states)

        # steering input, la is abr for lookahead
        horizon = 10 # 10 points lookahead
        horizon_id = min(self.waypt_id + horizon, len(self.waypts)-1) 

        wps = self.waypts[self.waypt_id:horizon_id]
        wp_next = wps[0]
        wp_pre  = self.waypts[self.waypt_id-1]

        pre_to_pos = np.array([current_pos[0]-wp_pre[0],
                            current_pos[1]-wp_pre[1]])
        
        direction = np.array([wp_next[0]-wp_pre[0], 
                            wp_next[1]-wp_pre[1]])
        
        projection = np.dot(pre_to_pos, direction)*direction/np.linalg.norm(direction)**2
        
        normal = pre_to_pos - projection
        lateral_err = np.linalg.norm(normal)

        cross_product = np.cross(direction, pre_to_pos)
        if cross_product > 0:
            lateral_err = -lateral_err

        trajectory_heading = math.atan2(
            wp_next[1] - wp_pre[1], 
            wp_next[0] - wp_pre[0])
        
        yaw_err = trajectory_heading - ego_state.theta
        yaw_err = wrap_angle(yaw_err)

        if previous_lat_control is not None:
            lateral_err_rate = (lateral_err - previous_lat_control.lateral_err) / self.dt
            yaw_err_rate = (yaw_err - previous_lat_control.yaw_err) / self.dt
        else:
            lateral_err_rate = 0.0
            yaw_err_rate = 0.0

        print(f"Yaw error: {yaw_err}, Lateral error: {lateral_err}")

        road_curve_vector = self.get_cr(wps)
        print(road_curve_vector)

        latcontrol_states = StatesLat(
            lateral_err,
            lateral_err_rate,
            yaw_err,
            yaw_err_rate,
            road_curve_vector
        )


        steering = self.steering(ego_state, latcontrol_states)
        print(f"acceleration: {acc}, steering: {steering}")

        return acc, steering, road_curve_vector[0]
        

    def get_cr(self, wps):
        N = len(wps)

        radius = np.zeros(N-2)
        for i in range(N-2):
            x1, y1 = wps[i]
            x2, y2 = wps[i+1] 
            x3, y3 = wps[i+2] 
            area = ((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))/2
            print(f"point1: {x1,y1}, point2: {x2,y2}, point3: {x3,y3}, area: {area}")
            if abs(area) < 10^(-6):
                radius[i] = np.inf
            else:
                a = two_point_distance(x1, y1, x2, y2)
                b = two_point_distance(x2, y2, x3, y3)
                c = two_point_distance(x1, y1, x3, y3)

                print(a, b, c)

                divider = math.sqrt(math.fabs((a+b+c)*(b+c-a)*(c+a-b)*(a+b-c)))
                radius[i] =  (a * b * c) / divider

        avg_radius = np.zeros(N-1)
        avg_radius[0] = radius[0]
        for i in range(1, N-2):
            avg_radius[i] = (radius[i-1] + radius[i]) / 2
        avg_radius[-1] = radius[-1]
        road_curve_vector = np.zeros(N-1)
        for i in range(N-1):
            if avg_radius[i] == np.inf:
                road_curve_vector[i] = 0
            else:
                road_curve_vector[i] = 1/avg_radius[i]

        return road_curve_vector

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
            new_id = self.waypt_id + 1
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

        
