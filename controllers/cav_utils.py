import numpy as np
from terasim_cosim.terasim_plugin.utils import sumo_to_utm_coordinate, utm_to_sumo_coordinate

class Vehicle:
    def __init__(self, vehicle_parameters:dict=None):
        if vehicle_parameters == None:
            self.mass = 1650.0
            self.Iz = 2315.0
            self.lr = 1.59
            self.lf = 1.11
            self.Cf = 133000
            self.Cr = 98800
            self.f0 = 24
            self.f1 = 19
        else:    
            self.mass = vehicle_parameters['mass'][0]
            self.Iz = vehicle_parameters['Iz'][0]
            self.lr = vehicle_parameters['lr'][0]
            self.lf = vehicle_parameters['lf'][0]
            self.Cr = vehicle_parameters['Cr'][0]
            self.Cf = vehicle_parameters['Cf'][0]
            self.f0 = vehicle_parameters['f0'][0]
            self.f1 = vehicle_parameters['f1'][0]

class State():
    def __init__(
            self, 
            x:float = 0, 
            y:float = 0, 
            z:float = 0,
            v_long:float = 0, 
            v_lat:float = 0, 
            theta:float = 0, 
            theta_dot:float = 0):
        
        self.x = x
        self.y = y
        self.z = z
        self.v_long = v_long
        self.v_lat = v_lat
        self.theta = theta
        self.theta_dot = theta_dot

    def get_state(self)->np.ndarray:
        return np.array([self.x, 
                         self.y, 
                         self.v_long, 
                         self.v_lat, 
                         self.theta,
                         self.theta_dot])

    def set_state(
            self,
            **kwargs):
        self.x = kwargs.get('x', self.x)
        self.y = kwargs.get('y', self.y)
        self.z = kwargs.get('z', self.z)
        self.v_long = kwargs.get('v_long', self.v_long)
        self.v_lat = kwargs.get('v_lat', self.v_lat)
        self.theta = kwargs.get('theta', self.theta)
        self.theta_dot = kwargs.get('theta_dot', self.theta_dot)

        return self
    
    def __add__(self, other):
        if isinstance(other, self.__class__):
            x = self.x + other.x
            y = self.y + other.y
            v_long = self.v_long + other.v_long
            v_lat = self.v_lat + other.v_lat
            theta = self.theta + other.theta
            theta_dot = self.theta_dot + other.theta_dot
        elif isinstance(other, np.ndarray):
            assert other.shape[0] == 6
            x = self.x + other[0]
            y = self.y + other[1]
            v_long = self.v_long + other[2]
            v_lat = self.v_lat + other[3]
            theta = self.theta + other[4]
            theta_dot = self.theta_dot + other[5]
        else: 
            raise TypeError("Unsupported operand type(s) for +: '{}' and '{}'").format(self.__class__, type(other))
        
        return State(x = x, y = y, v_long= v_long, v_lat=v_lat, theta=theta, theta_dot=theta_dot)
        

def sumo_states_to_utm_states(states:State):
    x, y = sumo_to_utm_coordinate(states.x, states.y)
    return State(x=x, 
                 y=y, 
                 v_long=states.v_long, 
                 v_lat = states.v_lat,
                 theta= states.theta,
                 theta_dot=states.theta_dot)

def utm_states_to_sumo_states(states:State):
    x, y = utm_to_sumo_coordinate(states.x, states.y)
    return State(x=x, 
                 y=y, 
                 v_long=states.v_long, 
                 v_lat = states.v_lat,
                 theta= states.theta,
                 theta_dot=states.theta_dot)