import unittest

from models.physical_class.grid_chunk import GridChunk
from models.ticking_class.ticking_earth import TickingEarth


class TestTickingEarth(unittest.TestCase):
    def setUp(self):
        self.earth = TickingEarth(shape=(10, 10))
        self.earth[0] = GridChunk.from_components_tuple((1000, 300, "WATER"), volume=1, index=0, parent=self.earth)
        self.earth[1] = GridChunk.from_components_tuple((1000, 250, "WATER"), volume=1, index=1, parent=self.earth)

    def test_temperature_balance(self):
        self.earth.update()
        self.assertEqual([299.5, 250.5], [x.temperature for x in self.earth.not_nones()])

    def test_temperature_balance_different_material(self):
        self.earth[1] = GridChunk.from_components_tuple((128, 250, "AIR"), volume=1, index=1, parent=self.earth)
        self.earth.update()
        self.assertAlmostEqual(266.1499505928854, self.earth[1].temperature)

    def test_temperature_balance_reach_equilibrium(self):
        self.earth[0] = GridChunk.from_components_tuple((1000, 300, "WATER"), volume=10, index=0, parent=self.earth)
        self.earth[1] = GridChunk.from_components_tuple((1000, 290, "WATER"), volume=10, index=1, parent=self.earth)
        for i in range(60 * int(1 / self.earth.get_universe().TIME_DELTA)):
            self.earth.update()
        self.assertAlmostEqual(self.earth[0].temperature, self.earth[1].temperature)
