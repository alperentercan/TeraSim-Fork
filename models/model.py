from models.bicycle_dynamics import BicycleDynamics
from  controllers.cav_utils import sumo_states_to_utm_states

class Model():

    def __init__(self, initial_state, dt=0.01, simulation_interval=0.04):
        integration_per_simulation_step = simulation_interval/dt
        assert integration_per_simulation_step.is_integer(), "Simulation interval must be an integer multiple of dt"

        self.state = initial_state
        self.dynamics = BicycleDynamics(vehicle_params =None,
                init_states=initial_state,
                dt = dt, 
                integrator='hybrid')
        

    def update_state(self):
        
