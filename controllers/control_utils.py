import numpy as np

def two_point_distance(x1,y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def wrap_angle(angle):
    while(angle > np.pi):
        angle -= np.pi*2
    while(angle < -np.pi):
        angle += np.pi*2
    return angle