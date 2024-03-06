import unittest

from models.physical_class.chunk_component import ChunkComponent
from models.physical_class.grid_chunk import GridChunk


class TestGridChunkBase(unittest.TestCase):
    def setUp(self):
        self.a_water_component = ChunkComponent(1000, 300, "Water")
        self.second_water_component = ChunkComponent(1000, 300, "Water")
        self.an_air_component = ChunkComponent(128, 300, "Air")
        self.a_land_component = ChunkComponent(1800, 300, "Land")
        self.second_land_component = ChunkComponent(145, 300, "Air")

        self.a_chunk = GridChunk([self.a_water_component], volume=1)

        self.grid_chunk_one = GridChunk([self.a_water_component, self.a_land_component], volume=1)
        self.grid_chunk_two = GridChunk([self.a_water_component, self.an_air_component], volume=1)
        self.grid_chunk_three = GridChunk([self.a_water_component, self.second_water_component], volume=1)

    def test_equality(self):
        self.assertNotEqual(self.grid_chunk_one, self.grid_chunk_three)

    def test_equality_other_type(self):
        self.assertNotEqual(self.grid_chunk_one, 25)

    def test_difference(self):
        self.assertNotEqual(self.grid_chunk_one, self.grid_chunk_two)

    def test_bracket_access(self):
        self.assertIs(self.a_chunk["WaTer"], self.a_chunk["WATER"],
                      "Bracket access can be used regardless of the capitalization")
        self.assertIs(self.a_chunk["water"], self.a_chunk["Water"],
                      "Bracket access can be used regardless of the capitalization")

    def test_access_variable(self):
        self.assertIs(self.a_chunk.water_component, self.a_chunk["WATER"],
                      "Variable access is equivalent to bracket access")

    def test_error_unknown_component(self):
        self.assertRaises(NotImplementedError, lambda: self.a_chunk.__setitem__("Chocolate", self.an_air_component))
