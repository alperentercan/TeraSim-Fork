import threading
import time
import numpy as np

import terasim_cosim.redis_msgs as redis_msgs
from terasim_cosim.constants import *
from terasim_cosim.redis_client_wrapper import create_redis_client

from models.bicycle_dynamics import State
from models.bicycle_dynamics import BicycleDynamics
import controllers.cav_utils as utils

class Model():

    def __init__(self, initial_state, dt=0.01, simulation_interval=0.04):
        self.simulation_interval = simulation_interval
        self.dt = dt
        self.integration_per_simulation_step = simulation_interval/dt
        assert self.integration_per_simulation_step.is_integer(), "Simulation interval must be an integer multiple of dt"

        self.state = initial_state
        self.dynamics = BicycleDynamics(vehicle_params =None,
                init_states=initial_state,
                dt = dt, 
                integrator='hybrid')

        key_value_config = {
            CAV_INFO: redis_msgs.ActorDict,
            VEHICLE_CONTROL: redis_msgs.VehicleControl
        }

        self.redis_client = create_redis_client(key_value_config=key_value_config)
        self.set_cav_info()

        self.running = False

    def update_state(self):
        control = self.get_control_commands()
        for _ in range(int(self.integration_per_simulation_step)):
            self.state = self.dynamics.next_state(self.state, control)
        self.set_cav_info()


    def _run_loop(self):
        while self.running:
            self.update_state()
            time.sleep(self.simulation_interval)

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_loop)
            self.thread.start()


    

    def get_control_commands(self):
        # For detailed fileds, see redis_msgs/VehicleControl.py
        veh_control = self.redis_client.get(VEHICLE_CONTROL)

        # Get control commands from the model
        if veh_control is None:
            veh_control_dict = {
            "timestamp": None,
            "brake_cmd": None,
            "throttle_cmd": 0,
            "steering_cmd": 0,
            "gear_cmd": None
        }
        else:
            veh_control_dict = {
                "timestamp": veh_control.header.timestamp,
                "brake_cmd": veh_control.brake_cmd,
                "throttle_cmd": veh_control.throttle_cmd,
                "steering_cmd": veh_control.steering_cmd,
                "gear_cmd": veh_control.gear_cmd
            }
        return veh_control_dict
        # return veh_control.header.timestamp, veh_control.brake_cmd, veh_control.throttle_cmd, veh_control.steering_cmd, veh_control.gear_cmd

    def set_cav_info(self):
        # For detailed files, see redis_msgs/VehicleDict.py
        cav_cosim_vehicle_info = redis_msgs.ActorDict()

        utm_current_state = utils.sumo_states_to_utm_states(self.state)

        # Set the timestamp
        cav_cosim_vehicle_info.header.timestamp = time.time()

        cav = redis_msgs.Actor()
        cav.length = 5.0
        cav.width = 1.8
        cav.height = 1.5

        cav.x = utm_current_state.x
        cav.y = utm_current_state.y
        cav.z = utm_current_state.z
        cav.orientation = utm_current_state.theta
        cav.speed_long = utm_current_state.v_long

        # Add cav to cav_cosim_vehicle_info
        cav_cosim_vehicle_info.data["CAV"] = cav

        self.redis_client.set(CAV_INFO, cav_cosim_vehicle_info)


def main():
    route_file = "controllers/highway_scenario/route.csv"
    with open(route_file, "r") as f:
        f.readline()
        waypts = [(float(line.split(',')[0]), float(line.split(',')[1])) for line in f.readlines()]

    x_init = waypts[0][0]
    y_init = waypts[0][1]
    z_init = 0
    theta_init = np.pi/2
    v_init = 0

    initial_state = State(
        x= x_init, 
        y= y_init,
        z= z_init,
        theta = theta_init,
        v_long = v_init)
        
    model = Model(initial_state)
    model.start()

if __name__ == "__main__":
    main()
