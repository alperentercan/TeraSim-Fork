import time
import numpy as np
import terasim_cosim.redis_msgs as redis_msgs

from terasim_cosim.constants import *
from terasim_cosim.redis_client_wrapper import create_redis_client

from terasim_cosim.terasim_plugin.utils import (
    send_user_av_control_wrapper,
    send_user_av_planning_wrapper,
)

from terasim_cosim.terasim_plugin.utils import sumo_to_utm_coordinate, utm_to_sumo_coordinate
# from av_decision_making_module.ozay_module.controller import Controller, DynamicBicycleModel
from cav_utils import *
from controller import Controller

def set_cav_info(redis_client, current_states, dt):
    # For detailed files, see redis_msgs/VehicleDict.py
    cav_cosim_vehicle_info = redis_msgs.ActorDict()

    # Set the timestamp
    cav_cosim_vehicle_info.header.timestamp = time.time()

    cav = redis_msgs.Vehicle()
    cav.length = 5.0
    cav.width = 1.8
    cav.height = 1.5

    cav.x = current_states.x
    cav.y = current_states.y
    cav.z = current_states.z
    cav.orientation = current_states.theta
    cav.speed_long = current_states.v_long

    # Add cav to cav_cosim_vehicle_info
    cav_cosim_vehicle_info.data["CAV"] = cav

    redis_client.set(CAV_INFO, cav_cosim_vehicle_info)

    time.sleep(dt)


def initialize_cav(init_states, dt):
    # Configure redis key-and data type
    key_value_config = {
        CAV_INFO: redis_msgs.ActorDict,
        TERASIM_ACTOR_INFO: redis_msgs.ActorDict,
    }

    redis_client = create_redis_client(key_value_config=key_value_config)

    # For detailed fileds, see redis_msgs/VehicleDict.py
    cav_cosim_vehicle_info = redis_msgs.ActorDict()
    # terasim_cosim_vehicle_info = redis_msgs.VehicleState()

    # Set the timestamp
    cav_cosim_vehicle_info.header.timestamp = time.time()
    # terasim_cosim_vehicle_info.header.timestamp = time.time()

    cav = redis_msgs.Actor()
    cav.length = 5.0
    cav.width = 1.8
    cav.height = 1.5

    cav.x = init_states.x
    cav.y = init_states.y
    cav.z = init_states.z
    cav.orientation = init_states.theta
    cav.speed_long = init_states.v_long

    # Add cav to cav_cosim_vehicle_info
    cav_cosim_vehicle_info.data["CAV"] = cav
    redis_client.set(CAV_INFO, cav_cosim_vehicle_info)
    print(f"Successfully set initial {CAV_INFO} list to redis!")
    time.sleep(dt)

    return redis_client


def main():
    # vehicle_params = './data/vehicle_params.csv'
    route_file = "./highway_scenario/route.csv"
    with open(route_file, "r") as f:
        f.readline()
        waypts = [(float(line.split(',')[0]), float(line.split(',')[1])) for line in f.readlines()]

    dt = 0.04
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
    
    redis_client = initialize_cav(utm_init_states, dt)

    model = Dynamics(vehicle_params =None,
                    init_states=initial_states,
                    dt = dt, 
                    integrator='hybrid')
    
    controller = Controller(waypts)
    counter = 1

    # for _ in range(1):
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

            model.states = current_state
            acc, steer = controller.compute_control(current_state, bvs)
            brake = -min(0, acc)/7.06
            throttle = max(0, acc)/2.87
            steer = steer/(0.4*np.pi)
            send_user_av_control_wrapper(brake, throttle, steer, 0)
            for i in range(20):
                current_state= model.update_states(acc, steer)

            true_current_state = State(
                x=current_state.x, 
                y=current_state.y, 
                theta = current_state.theta,
                v_long = current_state.v_long)
            
            print(f"Original true state: {true_current_state.v_long, current_state.v_long}")

            x_list = [current_state.x]
            y_list = [current_state.y]
            speed_list = [current_state.v_long]
            orientation_list = [current_state.theta]

            for iteration in range(int(2/0.025)):
                model.states = current_state
                acc, steer = controller.compute_control(current_state, bvs)

                current_state= model.update_states(acc, steer)
                # utm_current_state = sumo_states_to_utm_states(current_state)
                print(f"Waypoint ID at iteration {iteration}:  {controller.get_waypoint_id()}")
                x_list.append(current_state.x)
                y_list.append(current_state.y)
                speed_list.append(current_state.v_long)
                orientation_list.append(current_state.theta)

            model.states = true_current_state
            _, _ = controller.compute_control(true_current_state, bvs)
            print(f"Waypoint ID end of iteration:  {controller.get_waypoint_id()}")

            # send_user_av_planning_wrapper(x_list, y_list, speed_list, orientation_list)

            utm_current_state = sumo_states_to_utm_states(true_current_state)
            

            print(f"+++++++++++++++ Counter: {counter}")

            print(f"{round(100*counter/len(waypts),1)} completed", 
                "UTM", utm_current_state.x, utm_current_state.y, utm_current_state.v_long, 
                "SUMO", current_state.x, current_state.y, current_state.v_long
                )
            
            print(f"After true state: {true_current_state.v_long, current_state.v_long}")

            # set_cav_info(redis_client, utm_current_state, dt)
            counter = controller.get_waypoint_id()
            print(f"----------- Counter: {counter}")


if __name__ == "__main__":
    main()

