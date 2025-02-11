import csv
from math import sqrt, acos, degrees
import numpy as np

file_path = ".\coordinates\eval.csv"
file = open(file_path, "r")
reader = csv.reader(file)
eval_data = list(map(lambda q: (float(q[0]), float(q[1]), float(q[2])), reader))
ground_truth = []
distance_diff_milimeters = {"x": [], "z": []}
distances = []
degress_diff = []
origin = (0, 0, 0)

# generate ground truth for display size 250x250
for x in range(125, -126, -25):
    for z in range(-125, 126, 25):
        ground_truth.append((x, -500, z))

for i in range(len(eval_data)):
    x_gt, y_gt, z_gt = ground_truth[i]
    x_ev, y_ev, z_ev = eval_data[i]

    diff_x = abs(x_ev - x_gt)
    diff_z = abs(z_ev - z_gt)

    distance_diff_milimeters["x"].append(diff_x)
    distance_diff_milimeters["z"].append(diff_z)

    distance = sqrt(diff_x**2 + diff_z**2)
    distances.append(distance)

    vector1 = [x_gt - origin[0], y_gt - origin[1], z_gt - origin[2]]
    vector2 = [x_ev - origin[0], y_ev - origin[1], z_ev - origin[2]]

    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1] + vector1[2] * vector2[2]

    magnitude1 = sqrt(vector1[0]**2 + vector1[1]**2 + vector1[2]**2)
    magnitude2 = sqrt(vector2[0]**2 + vector2[1]**2 + vector2[2]**2)

    cosine_angle = dot_product / (magnitude1 * magnitude2)

    angle_rad = acos(cosine_angle)

    angle_deg = degrees(angle_rad)

    degress_diff.append(angle_deg)

# remove outliers that are 75mm or more off
for key, value in distance_diff_milimeters.items():
    distance_diff_milimeters[key] = list(filter(lambda q: q < 75, value))

# remove outliers from distance that are 75mm or more off
distances = list(filter(lambda q: q < 75, distances))

# remove outliers from degress that are 10° or more off
degress_diff = list(filter(lambda q: q < 10, degress_diff))

# print average distance difference
for key, value in distance_diff_milimeters.items():
    print(f"Average Distance Difference ({key}): {sum(value) / len(value):.2f}mm")

# print average distance
print(f"Average Distance: {sum(distances) / len(distances):.2f}mm")

# print accuracy degrees
print(f"Accuracy Degrees: {sum(degress_diff) / len(degress_diff):.2f}°")

data = np.array(degress_diff)
std_dev = np.std(data)

# print standard deviation degrees
print(f"Standard Deviation Degrees: {std_dev:.2f}°")