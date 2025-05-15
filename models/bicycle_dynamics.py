from  controllers.cav_utils import *

class BicycleDynamics:
    def __init__(self, vehicle_params, init_states, dt, integrator='hybrid'):
        self.v = Vehicle(vehicle_params)
        # state = init_states
        self.dt = dt
        self.integrator = integrator

    def next_state(self, current_state, control):
        acc, steer = control['throttle_cmd'], control['steering_cmd']
        # Vehicle dynamics equations
        if self.integrator == 'rk4':
            update_step = self.update_rk4(current_state, acc, steer)
        elif self.integrator == 'implicit':
            update_step = self.update_implicit(current_state, acc, steer)
        elif self.integrator == 'hybrid':
            if current_state.v_long > 6:
                update_step = self.update_rk4(current_state, acc, steer)
            else:
                update_step = self.update_implicit(current_state, acc, steer)
        else:
            print("Unknown integrator")
            raise RuntimeError
        
        new_state = current_state + self.dt*update_step
        return new_state

    def derivative(self, states, acc, steer):
        '''
        
        '''

        # necessary computations
        
        # front slip angle
        alpha_f = (
            steer 
            - np.arctan(states.v_lat + self.v.lf * states.theta_dot)
                        / (states.v_long + 1e-10)
                ) 
        # rear slip angle
        alpha_r = - np.arctan(
            (states.v_lat - self.v.lr * states.theta_dot)
              / (states.v_long + 1e-10)
            )  
        
        # lateral force at front tire
        Fyf = self.v.Cf * alpha_f  

        # lateral force at rear tire
        Fyr = self.v.Cr * alpha_r 

        # Calculate derivative components
        # longitudinal speed
        x_dot = (
            states.v_long*np.cos(states.theta) 
            - states.v_lat*np.sin(states.theta)
            )
        
        # lateral speed
        y_dot = (
            states.v_long*np.sin(states.theta) 
            + states.v_lat*np.cos(states.theta)
            )
        
        # longitudinal acceleration -- assuming small steering
        v_long_dot = (
            - self.v.f1/self.v.mass*states.v_long 
            - self.v.f0/self.v.mass + acc + states.v_lat*states.theta_dot 
            )
        
        #  lateral acceleration
        v_lat_dot = (
            (Fyf*np.cos(steer) + Fyr) / self.v.mass  
            - states.v_long*states.theta_dot
            )
        
        # rotational acceleration
        theta_dotdot = (self.v.lf*Fyf*np.cos(steer) - self.v.lr*Fyr)/self.v.Iz
        
        return np.array([
            x_dot, 
            y_dot, 
            v_long_dot, 
            v_lat_dot, 
            states.theta_dot, 
            theta_dotdot
            ])

    
    def update_rk4(self, state, acc, steer):
        # Update vehicle states
        k1 = self.derivative(state, acc, steer)
        k2 = self.derivative(state+k1*self.dt/2, acc, steer)
        k3 = self.derivative(state+k2*self.dt/2, acc, steer)
        k4 = self.derivative(state+k3*self.dt, acc, steer)

        return (1/6*k1 + 1/3*k2 + 1/3*k3 + 1/6*k4)

    def update_implicit(self, state, acc, steer):
        # Update the states based on the implicit Euler equation in arvix.org/pdf/2011.09612.pdf
        # x, y, v_lon, v_lat, yaw, r = self.x, self.y, self.v_lon, self.v_lat, self.yaw, self.r
        # print(f"Pre-update: {x=}, {y=}, {v_lon=}, {v_lat=}, {yaw=}, {r=}, {a=}, {delta=}{bcolors.ENDC}")
        # self.x = x + self.dt*(v_lon*math.cos(yaw) - v_lat*math.sin(yaw))
        # self.y = y + self.dt*(v_lat*math.cos(yaw) + v_lon*math.sin(yaw))
        # self.yaw = yaw + self.dt*r
        # self.v_lon = v_lon + self.dt*a
        # self.v_lat = (self.mass*v_lon*v_lat + self.dt*(self.lf*self.Cf - self.lr*self.Cr)*r - self.dt*self.Cf*delta*v_lon-self.dt*self.mass*v_lon**2*r)/(self.mass*v_lon - self.dt*(self.Cf+self.Cr))
        # self.r = (self.Iz*v_lon*r + self.dt*(self.lf*self.Cf - self.lr*self.Cr)*v_lat - self.dt*self.lf*self.Cf*delta*v_lon)/(self.Iz*v_lon-self.dt*(self.lf**2*self.Cf + self.lr**2*self.Cr))
        # print(f"Post update: {self.x=}, {self.y=}, {self.v_lon=}, {self.v_lat=}, {self.yaw=}, {self.r=}{bcolors.ENDC}")
        return self.derivative(state, acc, steer)



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