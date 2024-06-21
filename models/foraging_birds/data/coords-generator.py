# coords-generator.py
# (python coords-generator.py > coords.csv)

from itertools import product
from random import sample
import os

grid_size = 80
num_coords = 100

coords = sample(list(product(range(grid_size), repeat=2)), k=num_coords)

for coord in coords:
  print(str(coord[0]) + ',', coord[1])
