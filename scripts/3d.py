import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Text3D
import numpy as np

def visualizeRaycast(raycastEnd, cameraPos, cameraTarget, screenWidth = 250, screenHeight = 250, rayNumber = 1):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set limit for each axis
    ax.set_xlim(-250, 250)
    ax.set_ylim(0, 500)
    ax.set_zlim(-250, 250)

    # Set label for each axis
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Display
    r = 125
    x1 = r
    y1 = 500
    z1 = r
    x2 = - r
    y2 = 500
    z2 = - r
    verts = [(x1, y1, z1), (x2, y1, z1), (x2, y2, z2), (x1, y2, z2)]
    ax.add_collection3d(Poly3DCollection([verts], facecolors='gray', linewidths=1, edgecolors='r', alpha=.25))

    # Display label
    x, y, z = 130, 500, 130
    text = Text3D(x, y, z, 'Display', zdir='x')
    ax.add_artist(text)

    # Eye
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x3 = 10 * np.cos(u)*np.sin(v)
    y3 = 10 * np.sin(u)*np.sin(v)
    z3 = 10 * np.cos(v)
    ax.plot_wireframe(x3, y3, z3, color="gray", facecolors='gray')

    # Camera label
    x, y, z = 7, 0, 7
    text = Text3D(x, y, z, 'Eye', zdir='x')
    ax.add_artist(text)

    # Camera
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x3 = 10 * np.cos(u)*np.sin(v) + cameraPos[0]
    y3 = 10 * np.sin(u)*np.sin(v) + -1 * cameraPos[1]
    z3 = 10 * np.cos(v) + cameraPos[2]
    ax.plot_wireframe(x3, y3, z3, color="green", facecolors='green')

    # Camera label
    x, y, z = cameraPos[0] + 7, -1 * cameraPos[1], cameraPos[2] + 7
    text = Text3D(x, y, z, 'Camera', zdir='x')
    ax.add_artist(text)

    x_start, y_start, z_start = cameraPos[0], -1 * cameraPos[1], cameraPos[2]
    x_end, y_end, z_end = cameraTarget[0], cameraTarget[1], cameraTarget[2]
    ax.plot([x_start, x_end], [y_start, y_end], [z_start, z_end], color='green', linewidth=1)

    for i in range(rayNumber):
        x_start, y_start, z_start = 0, 0, 0
        x_end, y_end, z_end = raycastEnd[i]
        # Plot the line
        ax.plot([x_start, x_end], [y_start, y_end], [z_start, z_end], color='red')


    ax.view_init(elev=10, azim=-45)
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.margins(0,0,0)
    fig.set_size_inches(6.4, 4.8)
    
    plt.savefig('3d_plot.png', dpi=100)

if __name__ == "__main__":
    visualizeRaycast([(0, 500, 100)], (0, -50, 0), (0, 0, 0))