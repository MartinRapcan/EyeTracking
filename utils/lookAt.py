import numpy as np

# Define the makeLookAt function (assuming it is complete and correct)
def makeLookAt(position, target, up):
    forward = np.subtract(target, position) # create a vector pointing from the camera to the target
    forward = np.divide(forward, np.linalg.norm(forward)) # normalize the vector
    
    right = np.cross(forward, up) # create a vector pointing to the right of the camera
    right = np.divide(right, np.linalg.norm(right)) # normalize the vector

    up = np.cross(right, forward) # create a vector pointing up from the camera
    up = np.divide(up, np.linalg.norm(up)) # normalize the vector

    return np.array([[right[0], up[0], -forward[0], position[0]],
                     [right[1], up[1], -forward[1], position[1]],
                     [right[2], up[2], -forward[2], position[2]],
                     [0, 0, 0, 1]])

camera_up = np.array([0, 0, 1]) # Camera's up vector in global eye coordinates
cam_pos = np.array([20, -50, -10])  # Camera's position in global eye coordinates
eye_pos = np.array([0, 0, 0]) # Target's position in global eye coordinates

# Extract the rotation submatrix from the look-at matrix
lookat_matrix = makeLookAt(cam_pos, eye_pos, camera_up)
rot_matrix = lookat_matrix[:3, :3]

# Convert the rotation to Euler angles using the zxy convention
theta_z = np.arctan2(-rot_matrix[0, 1], rot_matrix[0, 0])
theta_x = np.arctan2(-rot_matrix[1, 2], rot_matrix[2, 2])
theta_y = np.arcsin(rot_matrix[0, 2])

# Convert the angles to degrees and print the result
euler_angles = np.array([theta_x, theta_y, theta_z]) * 180 / np.pi
print("Euler angles (in degrees): ", euler_angles)