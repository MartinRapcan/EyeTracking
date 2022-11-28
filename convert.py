import os
from PIL import Image

for file in os.listdir("dataset/data"):
    filename, extension  = os.path.splitext(file)
    if extension == ".PGM":
        new_file = f"{filename}.png"
        with Image.open(f"dataset/data/{file}") as im:
            im.save(f"dataset/data_png/{new_file}")