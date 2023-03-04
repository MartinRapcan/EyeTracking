import csv
import cv2
from random import randint, sample
from math import sqrt, atan2, cos, sin

input_path = "./coordinates/test.csv"
image_path = "./images/scanpath.png"

with open(input_path) as f:
	reader = csv.reader(f)
	raw = list(map(lambda q: (int(q[0]), int(q[1])), list(reader)))

raw = sample(raw, int(len(raw) * 0.1))

points_group = {}
threshold = 100

for i in range(0, len(raw)):
    x = raw[i][0]
    y = raw[i][1]
    for key in points_group:
         if abs(x - key[0]) <= threshold and abs(y - key[1]) <= threshold:
             points_group[key]['points'].append(raw[i])
             break
    else:
        points_group[(x, y)] = {'points': [raw[i]], 'middle': {'x': 0, 'y': 0}, 'diameter': 0}
    
    
diameter_scale = 0.1

for key in points_group:
    points = points_group[key]['points']
    x = 0
    y = 0
    diameter = 0

    for point in points:
        x += point[0]
        y += point[1]

    x = int(x / len(points))
    y = int(y / len(points))
    points_group[key]['middle']['x'] = x
    points_group[key]['middle']['y'] = y
    
    for point in points:
        distance = ((point[0] - x) ** 2 + (point[1] - y) ** 2) ** 0.5
        if distance == 0:
            distance = 10

        if distance > diameter:
            diameter = distance
    points_group[key]['diameter'] = int(diameter * diameter_scale * (len(points)% 10))

colors = {}
for key in points_group:
    colors[key] = (randint(50, 200), randint(50, 200), randint(50, 200))


image = cv2.imread(image_path)
overlay_circles = image.copy()
overlay_lines = image.copy()
points_group_keys = list(points_group)

for key in range(0, len(points_group) - 1):

    x1 = points_group[points_group_keys[key]]['middle']['x']
    y1 = points_group[points_group_keys[key]]['middle']['y']
    x2 = points_group[points_group_keys[key + 1]]['middle']['x']
    y2 = points_group[points_group_keys[key + 1]]['middle']['y']
    radius1 = points_group[points_group_keys[key]]['diameter']
    radius2 = points_group[points_group_keys[key + 1]]['diameter']

    distance = sqrt((x2 - x1)**2 + (y2 - y1)**2)

    if distance >= (radius1 + radius2):   
        angle = atan2(y2 - y1, x2 - x1)
        point1_x = x1 + radius1 * cos(angle)
        point1_y = y1 + radius1 * sin(angle)
        point2_x = x2 - radius2 * cos(angle)
        point2_y = y2 - radius2 * sin(angle)
        cv2.line(overlay_lines, (int(point1_x), int(point1_y)), (int(point2_x), int(point2_y)), colors[list(points_group)[key]], 2)
    
    cv2.circle(overlay_circles, (x1, y1), radius1, colors[points_group_keys[key]], 4)
    if key == len(points_group) - 2:
        cv2.circle(overlay_circles, (x2, y2), radius2, colors[points_group_keys[key + 1]], 4)


alpha_circles = 0.5
alpha_lines = 0.2

result = cv2.addWeighted(overlay_circles, alpha_circles, image, 1 - alpha_circles, 0)
result = cv2.addWeighted(overlay_lines, alpha_lines, result, 1 - alpha_lines, 0)

cv2.imwrite("./images/scanpath_test.png", result)