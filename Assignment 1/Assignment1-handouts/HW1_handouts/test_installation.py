import os
try :
    import gym_grid_driving
    from gym_grid_driving.envs.grid_driving import LaneSpec, MaskSpec, Point
except :
    print("Installation Not OK : Environment cannot be imported")
    exit()

if not os.path.isdir("/fast_downward/builds"):
    print("Installation Not OK : Planner not installed")
else :
    print("Installation OK : Environment and Planner present in the docker")
