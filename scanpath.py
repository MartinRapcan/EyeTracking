import csv
import cv2
from random import randint, sample
from math import sqrt, atan2, cos, sin

input_path = "./coordinates/test.csv"
image_path = "./images/scan_test.jpg"

image = cv2.imread(image_path)
overlay_circles = image.copy()
overlay_lines = image.copy()
alpha_circles = 0.6
alpha_lines = 0.2
TEXT_FACE = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.8
TEXT_THICKNESS = 2
TEXT_COLOR = (0, 0, 0)

points_group = {}
colors = {}
threshold = 150
order = 0

with open(input_path) as f:
	reader = csv.reader(f)
	raw = list(map(lambda q: (int(q[0]), int(q[1])), list(reader)))

raw = raw[:500]

main_point = None
for i in range(0, len(raw)):
    if not main_point:
        main_point = (raw[i][0], raw[i][1])

    if abs(raw[i][0] - main_point[0]) <= threshold and abs(raw[i][1] - main_point[1]) <= threshold:
        if not points_group.get(order):
            points_group[order] = {'points': [raw[i]], 'middle': {'x': 0, 'y': 0}, 'diameter': 0, 'index': order + 1}
        else:
            points_group[order]['points'].append(raw[i])
    
    else:
        order += 1
        main_point = (raw[i][0], raw[i][1])
        points_group[order] = {'points': [raw[i]], 'middle': {'x': 0, 'y': 0}, 'diameter': 0, 'index': order + 1}

points_group = dict(sorted(points_group.items(), key=lambda item: len(item[1]['points']), reverse=False))
points_group_keys = list(points_group)    

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
    
    
q1_index = int(len(points_group) * 0.25)
q2_index = int(len(points_group) * 0.5)
q3_index = int(len(points_group) * 0.75)

q1 = points_group_keys[:q1_index]
q2 = points_group_keys[q1_index:q2_index]
q3 = points_group_keys[q2_index:q3_index]
q4 = points_group_keys[q3_index:]

def setDiameter(array, diameter):
    for key in array:
        points_group[key]['diameter'] = diameter
        

setDiameter(q1, 30)
setDiameter(q2, 50)
setDiameter(q3, 70)
setDiameter(q4, 100)

points_group = dict(sorted(points_group.items(), key=lambda item: item[1]['index'], reverse=False))
points_group_keys = list(points_group)

color_step = 1 / len(points_group)
for key in points_group:
    shade = key * color_step
    colors[key] = (int(255 * shade), int(255 * shade), int(255 * shade))

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
    
    text_size, _ = cv2.getTextSize(str(points_group[points_group_keys[key]]['index']), TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
    text_origin = (int(x1 - text_size[0] / 2), int(y1 + text_size[1] / 2))
    
    cv2.circle(overlay_circles, (x1, y1), radius1, colors[points_group_keys[key]], -1)
    cv2.circle(overlay_circles, (x1, y1), radius1, (255, 255, 255), 3)
    cv2.putText(overlay_circles, str(points_group[points_group_keys[key]]['index']), text_origin, TEXT_FACE, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)
    if key == len(points_group) - 2:
        text_size, _ = cv2.getTextSize(str(points_group[points_group_keys[key]]['index']), TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
        text_origin = (int(x2 - text_size[0] / 2), int(y2 + text_size[1] / 2))
        cv2.circle(overlay_circles, (x2, y2), radius2, colors[points_group_keys[key + 1]], -1)
        cv2.circle(overlay_circles, (x2, y2), radius2, (255, 255, 255), 3)
        cv2.putText(overlay_circles, str(points_group[points_group_keys[key + 1]]['index']), text_origin, TEXT_FACE, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)


result = cv2.addWeighted(overlay_circles, alpha_circles, image, 1 - alpha_circles, 0)
result = cv2.addWeighted(overlay_lines, alpha_lines, result, 1 - alpha_lines, 0)

cv2.imwrite("./images/scan_test_after.jpg", result)