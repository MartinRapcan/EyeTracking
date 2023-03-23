import os
from PIL import Image
import cv2
from os.path import isfile, join
# generate ui pyside6-uic ./ui/main.ui > ui_mainwindow.py

for file in os.listdir("dataset/data"):
    filename, extension  = os.path.splitext(file)
    if extension == ".PGM":
        new_file = f"{filename}.png"
        with Image.open(f"dataset/data/{file}") as im:
            im.save(f"dataset/data_png/{new_file}")

def create_video(path, name, resolution):
    if resolution and path and name:
        images = [f for f in os.listdir(path) if isfile(join(path, f))]
        # i have one image in my folder make it like 1000 images
        images = images * 1000
        video = cv2.VideoWriter(f'dataset/{name}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, resolution)
        for image in images:
            img = cv2.imread(path + image)
            video.write(img)
        video.release()