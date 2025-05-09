from  controllers.cav_utils import *

class BicycleDynamics:
    def __init__(self, vehicle_params, init_states, dt, integrator='hybrid'):
        self.v = Vehicle(vehicle_params)
        self.states = init_states
        self.dt = dt
        self.integrator = integrator

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
    
    def set_current_state(self, derivatives):
        # x = self.states.x + self.dt*derivatives[0]
        # y = self.states.y + self.dt*derivatives[1]
        # v_long = self.states.v_long + self.dt*derivatives[2]
        # v_lat = self.states.v_lat + self.dt*derivatives[3]
        # theta = self.states.theta + self.dt*derivatives[4]
        # theta_dot = self.states.theta_dot + self.dt * derivatives[5]

        new_state = self.states + self.dt*derivatives
        # return State(x = x,
        #              y= y,
        #              v_long = v_long, 
        #              v_lat = v_lat, 
        #              theta= theta,
        #              theta_dot = theta_dot)
        return new_state

    def update_states(self, acc, steer):
        # Vehicle dynamics equations
        if self.integrator == 'rk4':
            update_step = self.update_rk4(acc, steer)
        elif self.integrator == 'implicit':
            update_step = self.update_implicit(acc, steer)
        elif self.integrator == 'hybrid':
            if self.states.v_long > 6:
                update_step = self.update_rk4(acc, steer)
            else:
                update_step = self.update_implicit(acc, steer)
        else:
            print("Unknown integrator")
            raise RuntimeError
        
        current_state = self.set_current_state(update_step)
        return current_state
    
    def update_rk4(self, acc, steer):
        # Update vehicle states
        k1 = self.derivative(self.states, acc, steer)
        k2 = self.derivative(self.states+k1*self.dt/2, acc, steer)
        k3 = self.derivative(self.states+k2*self.dt/2, acc, steer)
        k4 = self.derivative(self.states+k3*self.dt, acc, steer)

        return (1/6*k1 + 1/3*k2 + 1/3*k3 + 1/6*k4)

    def update_implicit(self, acc, steer):
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
        return self.derivative(self.states, acc, steer)

