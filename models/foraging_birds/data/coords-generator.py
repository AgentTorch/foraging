# coords-generator.py
# (python coords-generator.py > coords.csv)

from itertools import product
from random import sample
import os

grid_size = 10
num_coords = 40

coords = sample(list(product(range(grid_size), repeat=2)), k=num_coords)
print('x' + ',', 'y')
for coord in coords:
  print(str(coord[0]) + ',', coord[1])
