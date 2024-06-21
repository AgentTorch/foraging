# plot.py
# shows the birds on a scatterplot

import torch
import numpy as np
import osmnx as ox

import matplotlib
import matplotlib.pyplot as plotter
import matplotlib.patches as patcher
import contextily as ctx

from celluloid import Camera

class Plot:
  def __init__(self, max_x, max_y):
    # intialize the scatterplot
    self.figure, self.axes = None, None
    self.max_x, self.max_y = max_x, max_y

    plotter.xlim(0, max_x - 1)
    plotter.ylim(0, max_y - 1)
    self.i = 0

  def capture(self, state):
    graph = state['network']['agent_agent']['follower_birds']['graph']
    self.coords = [(node[1]['x'], node[1]['y']) for node in graph.nodes(data=True)]
    self.coords.sort(key=lambda x: -(x[0] + x[1]))

    if self.figure is None:
      self.figure, self.axes = ox.plot_graph(graph, edge_linewidth=0.3, edge_color='gray', show=False, close=False)
      ctx.add_basemap(self.axes, crs=graph.graph['crs'], source=ctx.providers.OpenStreetMap.Mapnik)
      self.axes.set_axis_off()

    # get coordinates of all the entities to show.
    bird = state['agents']['bird']

    # agar energy > 0 hai... toh zinda ho tum!
    alive_bird = bird['location']

    alive_bird_x, alive_bird_y = np.array([
      self.coords[(self.max_y * pos[0]) + pos[1]] for pos in alive_bird.long()
    ]).T

    # show prey in dark blue, predators in maroon, and
    # grass in light green.
    bird_scatter = self.axes.scatter(alive_bird_x, alive_bird_y , c='#0d52bd', marker='.')
    # grass_scatter = self.axes.scatter(grass_x, grass_y, c='#d1ffbd')

    # increment the step count.
    self.i += 1
    # show the current step count, and the population counts.
    self.axes.set_title('Boids Simulation', loc='left')
    self.axes.legend(handles=[
      patcher.Patch(color='#fc46aa', label=str(self.i) + ' step'),
      patcher.Patch(color='#0d52bd', label=str(len(alive_bird)) + ' bird'),
    ])

    # say cheese!
    self.figure.savefig('plots/boids-map-' + str(self.i) + '.png')

    # remove the points for the next update.
    bird_scatter.remove()