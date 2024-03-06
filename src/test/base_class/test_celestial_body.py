import unittest

from models.ABC.celestial_body import CelestialBody
from models.physical_class.earth import Earth
from models.physical_class.sun import Sun


class TestCelestialBody(unittest.TestCase):
    def setUp(self) -> None:
        self.earth = Earth(shape=(10, 10))
        self.sun = Sun()
        self.universe = self.earth.get_universe()

    def test_distance_earth_sun_constant(self):
        self.assertEqual(self.universe.distance_between(self.earth, self.sun), 1.496e11)

    def test_equal_distance_earth_sun_vice_versa(self):
        self.assertEqual(self.universe.distance_between(self.earth, self.sun),
                         self.universe.distance_between(self.sun, self.earth))

    def test_solid_angle_earth_sun(self):
        self.assertAlmostEqual(self.sun.solid_angle(self.earth), 5.71043e-09)

    def test_earth_sees_sun_vice_versa(self):
        self.assertTrue(self.sun.sees(self.earth))
        self.assertTrue(self.earth.sees(self.sun))

    def test_sees_not_implemented(self):
        self.assertFalse(self.sun.sees(CelestialBody(radius=10)))

    def test_all_objects_in_line_of_sight(self):
        self.universe.discover_everything()
        self.assertIn(self.earth, self.sun.objects_in_line_of_sight,
                      "After discovery, the earth can be seen from the sun")
        self.assertIn(self.sun, self.earth.objects_in_line_of_sight,
                      "After discovery, the sun can also be seen from the earth")
