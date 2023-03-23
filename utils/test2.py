import numpy as np
from scipy.spatial.transform import Rotation

def eulerToRot(theta, degrees=True) :
    r = Rotation.from_euler("zxy", (theta[2], theta[0], theta[1]), degrees)
    return r.as_matrix()
    
#camera coords to eye coords
if __name__ == "__main__":
    p = np.array([0, 0, -450]) #point in camera coord system
    
    r = eulerToRot([90, 0, 0]) #rotation of camera in eye coord system
    t = np.array([0, -50, 0]) #position of camera in eye coord system
    
    print(p @ r + t) #p transformed from camera coord system to eye coord system