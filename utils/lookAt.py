import numpy as np

def lookAt(camera, target, up):
    forward = target - camera
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    new_up = np.cross(right, forward)

    result = np.identity(4)
    result[0][0] = right[0]
    result[0][1] = right[1]
    result[0][2] = right[2]

    result[1][0] = new_up[0]
    result[1][1] = new_up[1]
    result[1][2] = new_up[2]

    result[2][0] = -forward[0]
    result[2][1] = -forward[1]
    result[2][2] = -forward[2]

    translation = np.identity(4)
    translation[0][3] = -camera[0]
    translation[1][3] = -camera[1]
    translation[2][3] = -camera[2]

    result = np.matmul(result, translation)
    return result


camera_up = np.array([0, 0, 1]) # Camera's up vector in global eye coordinates
cam_pos = np.array([0, -50, 0])  # Camera's position in global eye coordinates
eye_pos = np.array([0, 0, 0]) # Target's position in global eye coordinates

lookAt_matrix = lookAt(cam_pos, eye_pos, camera_up)
# Extract the rotation submatrix from the look-at matrix
rot_matrix = lookAt_matrix[:3, :3]

# Convert the rotation to Euler angles using the zxy convention
theta_z = np.arctan2(-rot_matrix[0, 1], rot_matrix[0, 0])
theta_x = np.arctan2(-rot_matrix[1, 2], rot_matrix[2, 2])
theta_y = np.arcsin(rot_matrix[0, 2])

# Convert the angles to degrees and print the result
euler_angles = np.array([theta_x, theta_y, theta_z]) * 180 / np.pi
print("Euler angles (in degrees): ", euler_angles)