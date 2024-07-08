# plot.py
# shows the prey and predators and grass on a scatterplot

import os
import torch
import numpy as np
import osmnx as ox

import matplotlib
import imageio
import os

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plotter
import matplotlib.patches as patcher
import contextily as ctx

class Plot:
    def __init__(self, max_x, max_y):
        # intialize the scatterplot
        self.figure, self.axes = None, None
        self.max_x, self.max_y = max_x, max_y
        self.images = []

        plotter.xlim(0, max_x - 1)
        plotter.ylim(0, max_y - 1)

    def capture(self, step, state):
        graph = state["network"]["agent_agent"]["follower_birds"]["graph"]
        self.coords = [(node[1]["x"], node[1]["y"]) for node in graph.nodes(data=True)]
        self.coords.sort(key=lambda x: -(x[0] + x[1]))

        if self.figure is None:
            self.figure, self.axes = ox.plot_graph(
                graph, edge_linewidth=0.3, edge_color="gray", show=False, close=False
            )
            ctx.add_basemap(
                self.axes,
                crs=graph.graph["crs"],
                source=ctx.providers.OpenStreetMap.Mapnik,
            )
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

        # show the current step count, and the population counts.
        self.axes.set_title("Birds Simulation", loc="left")
        self.axes.legend(
            handles=[
                patcher.Patch(color="#fc46aa", label=f"{step} step"),
                patcher.Patch(color="#0d52bd", label=f"{len(alive_bird)} birds"),
            ]
        )

        # say cheese!
        self.figure.savefig(f"plots/predator-prey-map-{step}.png")
        self.images.append(f"plots/predator-prey-map-{step}.png")

        # remove the points for the next update.
        bird_scatter.remove()
        # grass_scatter.remove()

    def compile(self, episode):
        if not os.path.exists('media/'):
            os.makedirs('media/')
        # convert all the images to a gif
        frames = [imageio.imread(f) for f in self.images]
        imageio.mimsave(f"media/predator-prey-{episode}.gif", frames, fps=20)

        # reset the canvas
        self.figure, self.axes = None, None
        self.images = []