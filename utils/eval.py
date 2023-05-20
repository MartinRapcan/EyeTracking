import csv

file_path = ".\coordinates\eval.csv"
file = open(file_path, "r")
reader = csv.reader(file)
eval_data = list(map(lambda q: (float(q[0]), float(q[1])), reader))
ground_truth = []
percentages = {"x": [], "y": []}
distance_diff_milimeters = {"x": [], "y": []}

# generate ground truth for display size 250x250
for x in range(125, -126, -25):
    for y in range(-125, 126, 25):
        ground_truth.append((x, y))

for i in range(len(eval_data)):
    x_gt, y_gt = ground_truth[i]
    x_ev, y_ev = eval_data[i]

    diff_x = abs(x_ev - x_gt)
    diff_y = abs(y_ev - y_gt)

    percentages["x"].append(diff_x / 250 * 100)
    percentages["y"].append(diff_y / 250 * 100)

    distance_diff_milimeters["x"].append(diff_x)
    distance_diff_milimeters["y"].append(diff_y)

# remove outliers that are 25% or more off
for key, value in percentages.items():
    percentages[key] = list(filter(lambda q: q < 25, value))

# remove outliers that are 75mm or more off
for key, value in distance_diff_milimeters.items():
    distance_diff_milimeters[key] = list(filter(lambda q: q < 75, value))

# print average percentage difference
for key, value in percentages.items():
    print(f"Average Percentage Difference ({key}): {sum(value) / len(value):.2f}%")

# print average distance difference
for key, value in distance_diff_milimeters.items():
    print(f"Average Distance Difference ({key}): {sum(value) / len(value):.2f}mm")
