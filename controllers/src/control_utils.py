import numpy as np

def two_point_distance(x1,y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def wrap_angle(angle):
    print(f"Wrapping angle: {angle}, {np.pi}")
    while(angle >= np.pi):
        print(f"Angle {angle} is greater than pi, wrapping")
        angle -= np.pi*2
    while(angle <= -np.pi):
        print(f"Angle {angle} is less than -pi, wrapping")
        angle += np.pi*2
    return angle