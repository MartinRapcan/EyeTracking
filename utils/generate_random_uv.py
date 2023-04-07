from random import uniform
import csv

def randomUVCoords(amount=200):
    uv_coords = []
    for i in range(amount):
        uv_coords.append((uniform(0, 1), uniform(0, 1)))

    with open('./coordinates/random_uv_coords.csv', 'w') as f:
        for uv in uv_coords:
            f.write(f'{uv[0]},{uv[1]}\n')

if __name__ == '__main__':
    randomUVCoords(60000)