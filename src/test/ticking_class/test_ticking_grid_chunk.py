import unittest

from models.physical_class.universe import Universe
from models.ticking_class.ticking_grid_chunk import TickingGridChunk


class TestTickingGridChunkBase(unittest.TestCase):

    @unittest.skipIf(TickingGridChunk.water_evaporation not in Universe.on_tick_methods,
                     "Not testing disabled on_tick method")
    def test_water_evaporation_no_air(self):
        water_chunk = TickingGridChunk.from_components_tuple((1000, 300, "Water"), volume=1)
        before_water_mass = water_chunk.water_component.mass
        water_chunk.update()
        evaporated_water = before_water_mass * Universe.TIME_DELTA * Universe.EVAPORATION_RATE
        self.assertEqual(water_chunk.water_component.mass, before_water_mass - evaporated_water)
        self.assertIsNotNone(water_chunk.air_component)

    @unittest.skipIf(TickingGridChunk.water_evaporation not in Universe.on_tick_methods,
                     "Not testing disabled on_tick method")
    def test_water_evaporation_with_air(self):
        water_air_chunk = TickingGridChunk.from_components_tuple((1000, 300, "Water"), (128, 300, "Air"), volume=1)
        before_water_mass = water_air_chunk.water_component.mass
        before_air_mass = water_air_chunk.air_component.mass

        water_air_chunk.update()

        evaporated_water = before_water_mass * Universe.TIME_DELTA * Universe.EVAPORATION_RATE

        self.assertEqual(water_air_chunk.water_component.mass, before_water_mass - evaporated_water)
        self.assertEqual(water_air_chunk.air_component.mass, before_air_mass + evaporated_water)
