import csv
import cv2
from math import sqrt, atan2, cos, sin

input_path = "./coordinates/uv_coords.csv"
image_path = "./images/scan_test.jpg"

image = cv2.imread(image_path)
image_width = image.shape[1]
image_height = image.shape[0]

def convert_uv_to_px(uv_data, width, height):
    return (int(uv_data[0] * width), int(uv_data[1] * height))

overlay_circles = image.copy()
overlay_lines = image.copy()
alpha_circles = 0.6
outline_width = 10
alpha_lines = 0.2
TEXT_FACE = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.8
TEXT_THICKNESS = 2
TEXT_COLOR = (0, 0, 0)

points_group = {}
colors = {}
threshold = 50
order = 0

with open(input_path) as f:
	reader = csv.reader(f)
	raw = list(map(lambda q: (float(q[0]), float(q[1])), reader))

for i in range(0, len(raw)):
    raw[i] = convert_uv_to_px(raw[i], image_width, image_height)

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
    points_group[key]['diameter'] = 20
    

different_lengths = {}
for key in points_group:
    if not different_lengths.get(len(points_group[key]['points'])):
        different_lengths[len(points_group[key]['points'])] = [key]
    else:
        different_lengths[len(points_group[key]['points'])].append(key)

diameters = {1: 20, 2: 40, 3: 60, 4: 80}
if len(different_lengths) == 1:
    for key in points_group:
        points_group[key]['diameter'] = diameters[1]
elif len(different_lengths) == 2:
    for key in different_lengths:
        for index in different_lengths[key]:
            points_group[index]['diameter'] = diameters[key]
elif len(different_lengths) == 3:
    for key in different_lengths:
        for index in different_lengths[key]:
            points_group[index]['diameter'] = diameters[key]
elif len(different_lengths) == 4:
    for key in different_lengths:
        for index in different_lengths[key]:
            points_group[index]['diameter'] = diameters[key]
else:
    first_key = different_lengths[list(different_lengths)[0]]
    last_key = different_lengths[list(different_lengths)[-1]]
    remaining_keys = list(different_lengths)[1:-1]
    first_middle_keys = remaining_keys[:len(remaining_keys) // 2]
    last_middle_keys = remaining_keys[len(remaining_keys) // 2:]
    for key in first_key:
        points_group[key]['diameter'] = diameters[1]
    for key in last_key:
        points_group[key]['diameter'] = diameters[4]
    for key in first_middle_keys:
        points_group[key]['diameter'] = diameters[2]
    for key in last_middle_keys:
        points_group[key]['diameter'] = diameters[3]
        

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
        point1_x = x1 + (radius1 + outline_width // 2) * cos(angle)
        point1_y = y1 + (radius1 + outline_width // 2) * sin(angle)
        point2_x = x2 - (radius2 + outline_width // 2) * cos(angle)
        point2_y = y2 - (radius2 + outline_width // 2) * sin(angle)
        cv2.line(overlay_lines, (int(point1_x), int(point1_y)), (int(point2_x), int(point2_y)), colors[list(points_group)[key]], 4)
    
    text_size, _ = cv2.getTextSize(str(points_group[points_group_keys[key]]['index']), TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
    text_origin = (int(x1 - text_size[0] / 2), int(y1 + text_size[1] / 2))
    
    #cv2.circle(overlay_circles, (x1, y1), radius1, colors[points_group_keys[key]], -1)
    cv2.circle(overlay_circles, (x1, y1), radius1, colors[points_group_keys[key]], outline_width)
    #cv2.putText(overlay_circles, str(points_group[points_group_keys[key]]['index']), text_origin, TEXT_FACE, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)
    if key == len(points_group) - 2:
        text_size, _ = cv2.getTextSize(str(points_group[points_group_keys[key]]['index']), TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
        text_origin = (int(x2 - text_size[0] / 2), int(y2 + text_size[1] / 2))
        #cv2.circle(overlay_circles, (x2, y2), radius2, colors[points_group_keys[key + 1]], -1)
        cv2.circle(overlay_circles, (x2, y2), radius2, colors[points_group_keys[key + 1]], outline_width)
        #cv2.putText(overlay_circles, str(points_group[points_group_keys[key + 1]]['index']), text_origin, TEXT_FACE, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)


result = cv2.addWeighted(overlay_circles, alpha_circles, image, 1 - alpha_circles, 0)
result = cv2.addWeighted(overlay_lines, alpha_lines, result, 1 - alpha_lines, 0)

# TODO: generate multiple images with different types of scanpath
cv2.imwrite("./images/scan_test_after.jpg", result)