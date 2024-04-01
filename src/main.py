import random
import sys

from PyQt5 import QtWidgets
from tqdm import trange

from controller.main_controller import MainController
#from models.physical_class.grid_chunk import GridChunk
from models.physical_class.universe import Universe
from models.ticking_class.ticking_earth import TickingEarth
from models.ticking_class.ticking_sun import TickingSun

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "GUI":
        app = QtWidgets.QApplication([])
        controller = MainController()

        controller.view.show()
        app.exec_()
    else:
        backend = "numpy"
        universe = Universe()
        universe.sun = TickingSun()  # Can be replaced with Sun()
        print("Generating the earth...")
        universe.earth = TickingEarth(shape=(40, 40, 40), backend=backend)  # Use 1 as the z dimension if you want a 2D grid
        universe.discover_everything()

        # Fills the earth with random GridChunk of water
        filling_density = 1
        universe.earth.fill_with_water()

        print("Done.")
        print("Updating Universe 10 times...")
        print(universe)
        for i in trange(10):
            universe.update_all()
        print(universe)
