import time
import numpy as np
import terasim_cosim.redis_msgs as redis_msgs

from terasim_cosim.constants import *
from terasim_cosim.redis_client_wrapper import create_redis_client

from terasim_cosim.terasim_plugin.utils import sumo_to_utm_coordinate, utm_to_sumo_coordinate
# from av_decision_making_module.ozay_module.controller import Controller, DynamicBicycleModel
from controllers.cav_utils import *
from models.bicycle_dynamics import BicycleDynamics
from controllers.controller import Controller
from terasim_cosim.terasim_plugin.utils import (
    send_user_av_control_wrapper,
    send_user_av_planning_wrapper,
)


def main():
    # vehicle_params = './data/vehicle_params.csv'
    route_file = "controllers/highway_scenario/route.csv"
    with open(route_file, "r") as f:
        f.readline()
        waypts = [(float(line.split(',')[0]), float(line.split(',')[1])) for line in f.readlines()]

    dt = 0.01
    x_init = waypts[0][0]
    y_init = waypts[0][1]
    z_init = 0
    theta_init = np.pi/2
    v_init = 0

    initial_states = State(
        x= x_init, 
        y= y_init,
        z= z_init,
        theta = theta_init,
        v_long = v_init)
    
    utm_init_states = sumo_states_to_utm_states(initial_states)
    print(utm_init_states.x, utm_init_states.y)
    
    key_value_config = {
        CAV_INFO: redis_msgs.ActorDict,
        TERASIM_ACTOR_INFO: redis_msgs.ActorDict,
    }

    redis_client = create_redis_client(key_value_config=key_value_config)

    model = BicycleDynamics(vehicle_params =None,
                    init_states=initial_states,
                    dt = dt, 
                    integrator='hybrid')
    
    controller = Controller(waypts)
    counter = 1

    while counter < len(waypts):
        cav_info = redis_client.get(CAV_INFO)
        bv_info = redis_client.get(TERASIM_ACTOR_INFO)
        # bvs = [[bv.x, bv.y, bv.speed_long] for bv in bv_info.data]
        if bv_info:
            bvs = [ utm_states_to_sumo_states(
                State(
                    x = bv_info.data[bv_name].x,
                    y= bv_info.data[bv_name].y,
                    v_long = bv_info.data[bv_name].speed_long,
                    theta = bv_info.data[bv_name].orientation) 
                    )
                    for bv_name in bv_info.data
                ]
        else:
            bvs = []
        # [print(bv.x, bv.y, bv.v_long) for bv in bvs]
        if cav_info:
            cav = cav_info.data["CAV"]
            cav_state = State(
                x=cav.x, 
                y=cav.y, 
                theta = cav.orientation,
                v_long = cav.speed_long)
            current_state = utm_states_to_sumo_states(cav_state)
            acc, steer = controller.compute_control(current_state, bvs)

            brake = -min(0, acc)/7.06
            throttle = max(0, acc)/2.87
            steer = steer/(0.4*np.pi)

            print("acc, steer", acc, steer)
            print("brake, throttle, steer", brake, throttle, steer)

            send_user_av_control_wrapper(brake, throttle, steer, 0)

            # #  for i in range(20):
            # current_state= model.next_state(current_state, {"throttle_cmd":acc, "steering_cmd":steer})
            # # utm_current_state = sumo_states_to_utm_states(current_state)
            
            # print(f"{round(100*counter/len(waypts),1)} completed", 
            #     "UTM", utm_current_state.x, utm_current_state.y, utm_current_state.v_long, 
            #     "SUMO", current_state.x, current_state.y, current_state.v_long
            #     )
            # counter = controller.get_waypoint_id()


if __name__ == "__main__":
    main()

