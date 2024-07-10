import random
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


from tqdm import trange

from models.physical_class.universe import Universe
from models.ticking_class.ticking_earth import TickingEarth
from models.ticking_class.ticking_sun import TickingSun

np.random.seed(0)


def init_graph():
    fig, ax = plt.subplots()
    # Do not plot the borders
    cax = ax.matshow(universe.earth.chunk_temp[1:-1,1:-1,20], cmap='coolwarm')
    fig.colorbar(cax)
    plt.savefig("initial_plot.png")
    plt.ion()
    return fig, ax, cax


def update_graph(cax):
    # Update the colorbar
    cax.set_norm(Normalize(vmin=np.min(universe.earth.chunk_temp[1:-1,1:-1,20]), vmax=np.max(universe.earth.chunk_temp[1:-1,1:-1,20])))
    # Update the data
    cax.set_array(universe.earth.chunk_temp[1:-1,1:-1,20])
    plt.draw()
    plt.pause(0.1)


def final_plot():
    plt.ioff()
    plt.savefig(f"final_plot_{nb_steps}_steps.png")
    plt.show()

if __name__ == "__main__":
    backend = "numpy"
    grid_shape = (50, 50, 80)
    nb_steps = 50

    # Set to True to plot an interactive evolution of the temperature, False to compute at full speed
    # Note that the plot will be updated every 0.1 seconds, slowing down the simulation in order to visualize it
    visualisation = False


    universe = Universe()
    universe.sun = TickingSun()  # Can be replaced with Sun()
    print("Running model with backend:", backend)
    print("Generating the earth...")
    universe.earth = TickingEarth(shape=grid_shape, backend=backend)
    universe.discover_everything()

    # Fills the earth with random GridChunk of water
    universe.earth.fill_with_water()

    print("Done.")
    print("Updating Universe 10 times...")
    print(universe)
    

    if visualisation:
        fig, ax, cax = init_graph() # Uncomment this line to plot the evolution of the temperature

    for i in trange(nb_steps):
        universe.update_all()
        if visualisation:
            update_graph(cax) 

    print(universe)
    if visualisation:
        final_plot() # Uncomment this line to plot the evolution of the temperature


