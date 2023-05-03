import cv2
from pupil_detectors import Detector2D
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode
import os
from matplotlib import pyplot
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Text3D
import numpy as np
from scipy.spatial.transform import Rotation
import sys

def intersectPlane(n, p0, l0, l):
    denom = matmul(-n, l)
    if (denom > sys.float_info.min):
        p0l0 = p0 - l0
        t = matmul(p0l0, -n) / denom
        return t
    return -1.0
    
def matmul(v1, v2, pad=False, padBy=1.0):
    if(pad is True):
        return np.matmul(v1, np.append(v2, padBy))[:-1]
    return np.matmul(v1, v2)
    
def getPoint(ray, distance):
    return ray[0] + ray[1] * distance

def normalize(v):
    return v / magnitude(v)
    
def magnitude(v):
    return np.sqrt(sqrMagnitude(v))
        
def sqrMagnitude(v):
    return matmul(v, v)

def eulerToRot(theta, degrees=True) :
    r = Rotation.from_euler("zxy", (theta[2], theta[0], theta[1]), degrees)
    return r.as_matrix()
    
def transform(p, position, rotMat):
    return rotate(p, rotMat) + position
    
def inverseTransform(p, position, rotMat):
    return (p - position) @ rotMat #inverse rotation
    
def rotate(p, rotMat):
    return p @ rotMat.T
    
config = {
    "detector_2d":{
        "intensity_range":32,
        "pupil_size_max":120,
        "pupil_size_min":50,
    },
    "detector_3d":{
    },
    "camera": {
        "focal_length":772.55,
        "resolution": (640, 480)
    },
    "refraction_correction":False
}

def plotSphere(ax, pos, scale=8, color="gray"):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x3 = scale * np.cos(u)*np.sin(v) + pos[0]
    y3 = scale * np.sin(u)*np.sin(v) + pos[1]
    z3 = scale * np.cos(v) + pos[2]
    ax.plot_wireframe(x3, y3, z3, color=color, facecolors=color)
    return

def visualize(ax, cameraPos, eyePos, cameraDirs, gazeDir, screenWidth = 250, screenHeight = 250):
    # Set limit for each axis
    ax.set_xlim(250, -250)
    ax.set_ylim(0, -500)
    ax.set_zlim(-250, 250)

    # Set label for each axis
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Display
    x1 = screenWidth / 2
    y1 = -500
    z1 = screenHeight / 2
    x2 = - screenWidth / 2
    y2 = -500
    z2 = - screenHeight / 2
    verts = [(x1, y1, z1), (x2, y1, z1), (x2, y2, z2), (x1, y2, z2)]
    ax.add_collection3d(Poly3DCollection([verts], facecolors='gray', linewidths=1, edgecolors='r', alpha=.25))

    # Display label
    x, y, z = 130, -500, 130
    text = Text3D(x, y, z, 'Display', zdir='x')
    ax.add_artist(text)

    # Eye
    plotSphere(ax, eyePos, color="gray")

    # Eye label
    x, y, z = 7, 0, 7
    text = Text3D(x, y, z, 'Eye', zdir='x')
    ax.add_artist(text)

    # Camera
    plotSphere(ax, cameraPos, color="green")
    
    # Origin
    plotSphere(ax, (0, 0, 0), color="black", scale=2)

    # Camera label
    x, y, z = cameraPos[0] + 7, cameraPos[1], cameraPos[2] + 7
    text = Text3D(x, y, z, 'Camera', zdir='x')
    ax.add_artist(text)

    # Camera axes
    ax.quiver(*cameraPos, *cameraDirs[0], length=25, normalize=True, color='red')
    ax.quiver(*cameraPos, *cameraDirs[1], length=25, normalize=True, color='green')
    ax.quiver(*cameraPos, *cameraDirs[2], length=25, normalize=True, color='blue')
    
    origin = (0, 0, 0)
    dirs = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    # Camera axes
    ax.quiver(*origin, *dirs[0], length=25, normalize=True, color='red')
    ax.quiver(*origin, *dirs[1], length=25, normalize=True, color='green')
    ax.quiver(*origin, *dirs[2], length=25, normalize=True, color='blue')
    
    # Gaze ray
    ax.quiver(*eyePos, *gazeDir, length=100, normalize=True, color='purple')
    
    return
    
def plot(ax, points):
    ax.scatter3D(*np.array(points).T, color = "purple", s=2)
    return

if __name__ == "__main__":

    cameraPos = np.array([20, -50, -10])
    #cameraRot = np.array([-59.52578847, -63.43495128,  66.68740363])
    #cameraRotMat = eulerToRot(cameraRot)
    cameraRotMat = np.array([
        [0.884918212890625, -0.105633445084095, -0.4536091983318329],
        [0.4657464325428009, 0.20070354640483856, 0.8618574738502502],
        [0.0, -0.973940372467041, 0.22680459916591644]
    ])
    displaySize = (250, 250) #width, height
    displayPos = np.array([0, -500, 0])
    displayRot = np.array([0, 0, 180])
    displayRotMat = eulerToRot(displayRot)
    displayNormalLocal = np.array([0, -1, 0])
    displayNormalWorld = normalize(rotate(displayNormalLocal, displayRotMat))
    
    cameraDirsWorld = (
        rotate(np.array((1, 0, 0)), cameraRotMat),
        rotate(np.array((0, 1, 0)), cameraRotMat),
        rotate(np.array((0, 0, 1)),cameraRotMat)
    )
    
    #init test img
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    image = cv2.putText(image, 'Test', (320, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)

    detector_2d = Detector2D()
    detector_2d.update_properties(config["detector_2d"])
    
    camera = CameraModel(**config["camera"])
    
    detector_3d = Detector3D(camera=camera, long_term_mode=DetectorMode.blocking)
    detector_3d.update_properties(config["detector_3d"])

    img_dir = ".\dataset-Vincur\latest"

    pyplot.ion()
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.tight_layout()

    #model warmup
    for i in range(121):
        frame = cv2.imread(os.path.join(img_dir, f"example_{i}.png"))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result_2d = detector_2d.detect(gray, frame)
        result_2d["timestamp"] = i
        result_3d = detector_3d.update_and_detect(result_2d, gray)
    
    gazePointsWorld = []
    #get data
    for i in range(121):
        frame = cv2.imread(os.path.join(img_dir, f"example_{i}.png"))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        result_2d = detector_2d.detect(gray, frame)
        result_2d["timestamp"] = i
        
        result_3d = detector_3d.update_and_detect(result_2d, gray, apply_refraction_correction=config["refraction_correction"])
        
        ax.clear()
        
        #camera system to world
        eyePosWorld = transform(np.array(result_3d["sphere"]["center"]), cameraPos, cameraRotMat)
        gazeRay = normalize(rotate(result_3d["circle_3d"]["normal"], cameraRotMat))
        visualize(ax, cameraPos, eyePosWorld, cameraDirsWorld, gazeRay, *displaySize)
        
        intersectionTime = intersectPlane(displayNormalWorld, displayPos, eyePosWorld, gazeRay)
        if (intersectionTime > 0.0):
            planeIntersection = getPoint([eyePosWorld, gazeRay], intersectionTime)
            gazePointsWorld.append(planeIntersection)
            plot(ax, gazePointsWorld)
            
            planeIntersectionDisplayLocal = inverseTransform(planeIntersection, displayPos, displayRotMat)
            u = planeIntersectionDisplayLocal[0] / displaySize[0] + 0.5
            v = 1 - (planeIntersectionDisplayLocal[2] / displaySize[1] + 0.5)
            x = int(u * image.shape[1])
            y = int(v * image.shape[0])
            if x > 0 and x < image.shape[1]:
                if y > 0 and y < image.shape[0]:
                    cv2.circle(image, (x, y), 4, (128, 0, 128), 2)
            
        cv2.imshow("Image", image)
        cv2.imshow("2d detection", frame)
        
        if cv2.waitKey(0) == ord('q'):
            break
    
    
        
    cv2.destroyAllWindows()