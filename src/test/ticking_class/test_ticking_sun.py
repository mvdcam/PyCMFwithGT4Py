import math
import unittest

from models.physical_class.universe import Universe
from models.ticking_class.ticking_earth import TickingEarth
from models.ticking_class.ticking_grid_chunk import TickingGridChunk
from models.ticking_class.ticking_sun import TickingSun


class TestTickingUniverse(unittest.TestCase):
    @unittest.skipIf(TickingSun.radiate_energy_outwards not in Universe.on_tick_methods,
                     "Not testing disabled on_tick method")
    def test_sun_radiate_earth(self):
        earth = TickingEarth(shape=(10, 1))
        earth[0] = TickingGridChunk.from_components_tuple((1000, 0, "WATER"), volume=1, index=0, parent=earth)
        sun = TickingSun()
        # Change the energy radiated per second to match a radiation of exactly 1 Joule per time delta towards the earth
        sun.energy_radiated_per_second = (4 * math.pi) / (sun.solid_angle(earth) * Universe.TIME_DELTA)
        sun.get_universe().discover_everything()
        before_energy = earth.compute_total_energy()
        sun.update()
        self.assertAlmostEqual(earth.compute_total_energy(), before_energy + 1)
