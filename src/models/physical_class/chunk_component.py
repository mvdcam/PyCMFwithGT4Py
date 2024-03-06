import math
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models.physical_class.grid_chunk import GridChunk
import constants


class ChunkComponent:
    """
    Standalon Third layer implementation
    A class to represent a component of a chunk.

    ...

    Attributes
    ----------
    chunk: GridChunk
        The reference to the parent (container) GridChunk
    mass: float
        The Mass of the component, will be used as kg
    specific_heat_capacity:
        The amount of heat that must be added to a material to increase its temperature
        See https://en.wikipedia.org/wiki/Specific_heat_capacity
        Expressed in Joules per Kilograms per Kelvin (Water = 4184 Jkg-1K-1)
    heat_transfer_coefficient:
        Coefficient for Newton's law of Cooling
        See https://en.wikipedia.org/wiki/Heat_transfer_coefficient
        Expressed in Watt per Meter squared per Kelvin
    energy: float
        The molecular kinetic energy of the component. It is used to store and compute the temperature of the
        component
        """
    chunk: "GridChunk" = None
    mass: float
    energy: float = 0
    specific_heat_capacity: float
    heat_transfer_coefficient: float

    def __init__(self, mass: float, temperature: float, component_type: str):
        # Information on the component
        self.type = component_type.upper()

        # Physical properties
        self.mass = mass

        self.specific_heat_capacity = constants.SPECIFIC_HEAT_CAPACITY[self.type]
        self.heat_transfer_coefficient = constants.HEAT_TRANSFER_COEFFICIENT[self.type]
        self.__set_temperature(temperature)

    def __eq__(self, other: Optional["ChunkComponent"]):
        if other is None:
            return False
        return math.isclose(self.mass, other.mass) and math.isclose(self.energy, other.energy)

    @classmethod
    def init_default(cls, *, component_type: str):
        return cls(mass=1000, temperature=300, component_type=component_type)

    def change_mass(self, value: float):
        """
        Change the mass of the component and keep the same temperature
        :param value:
        :return:
        """
        temperature = self.__get_temperature()  # Save the temperature, because the energy will remain the same in smaller mass
        self.mass = value
        self.__set_temperature(temperature)

    def __get_temperature(self):
        return self.energy / (self.specific_heat_capacity * self.mass)

    def __set_temperature(self, temperature: float):
        self.energy = self.specific_heat_capacity * self.mass * temperature

    def __str__(self):
        res = f"{self.type} Component \n" \
              f"Mass : {self.mass} \n " \
              f"Temperature : {self.__get_temperature()} K ({self.energy} J)"
        return res

    def deep_copy(self):
        return ChunkComponent(self.mass, self.__get_temperature(), component_type=self.type)

    def is_empty(self):
        return math.isclose(self.mass, 0)
