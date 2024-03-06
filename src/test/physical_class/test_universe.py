import unittest

from models.ABC.celestial_body import CelestialBody
from models.physical_class.earth import Earth
from models.physical_class.universe import Universe


class TestUniverse(unittest.TestCase):
    def test_universe_setup_when_accessing(self):
        self.assertIsInstance(CelestialBody.get_universe(), Universe, "First access return the instance")
        self.assertIsNotNone(CelestialBody._CelestialBody__universe,
                             "Universe is properly setup and ready to be used by other models")

    def test_universe_universal_reference(self):
        self.assertIs(CelestialBody.get_universe(), Earth.get_universe())
