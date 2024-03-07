from models.ABC.celestial_body import CelestialBody
from models.base_class.earth_base import EarthBase
from models.physical_class.grid_chunk import GridChunk
import constants


import numpy as np
from gt4py.cartesian import gtscript
from gt4py.cartesian.gtscript import PARALLEL, BACKWARD, computation, interval, IJ, IJK, Field
import gt4py.storage as gt_storage
import typing

Field3D = gtscript.Field[np.float64]


class Earth(EarthBase, CelestialBody):
    """
    Second layer of the Earth model.
    In this layer are all the physical properties and functions of the Earth implemented. It is here that we will add
    new model variables
    """
    albedo: float = 0.3
    CARBON_EMISSIONS_PER_TIME_DELTA: float = 1_000_000  # ppm
    backend: str

    def __init__(self, shape: tuple, radius: float = 6.3781e6, *, parent=None, backend="numpy"):
        EarthBase.__init__(self, shape, parent=parent, backend=backend)
        CelestialBody.__init__(self,
                               radius)  # The default radius of the earth was found here https://arxiv.org/abs/1510.07674
        self.get_universe().earth = self
        self.get_universe().discover_everything()
        self.backend = backend

        @gtscript.function
        def component_ratio(component_mass: gtscript.Field[float], chunk_mass: gtscript.Field[float]) -> float:
            return component_mass[0, 0, 0] / chunk_mass[0, 0, 0]
        
        def compute_heat_transfer_coefficient(water_mass: gtscript.Field[float], 
                                              air_mass: gtscript.Field[float], 
                                              land_mass: gtscript.Field[float], 
                                              chunk_mass: gtscript.Field[float],
                                              heat_transfer_coefficient: gtscript.Field[float]):
            with computation(PARALLEL), interval(...):
                heat_transfer_coefficient = component_ratio(water_mass, chunk_mass) * constants.WATER_HEAT_TRANSFER_COEFFICIENT + \
                                             component_ratio(air_mass, chunk_mass) * constants.AIR_HEAT_TRANSFER_COEFFICIENT + \
                                                component_ratio(land_mass, chunk_mass) * constants.LAND_HEAT_TRANSFER_COEFFICIENT
                
        def compute_chunk_composition(water_mass: gtscript.Field[float], 
                                      air_mass: gtscript.Field[float], 
                                      land_mass: gtscript.Field[float], 
                                      chunk_mass: gtscript.Field[float],
                                      water_composition: gtscript.Field[float],
                                      air_composition: gtscript.Field[float],
                                      land_composition: gtscript.Field[float]):
            with computation(PARALLEL), interval(...):
                water_composition = component_ratio(water_mass, chunk_mass)
                air_composition = component_ratio(air_mass, chunk_mass)
                land_composition = component_ratio(land_mass, chunk_mass)

        @gtscript.function
        def temperature_to_energy(temperature: gtscript.Field[float], mass: gtscript.Field[float]) -> float:
            """
            Set the temperature of the component by computing the energy from the mass and the temperature
            :param temperature:
            :return:
            """
            return temperature[0, 0, 0] * mass[0, 0, 0] * constants.WATER_HEAT_CAPACITY
        
        def temperature_to_energy_field(temperature: gtscript.Field[float], mass: gtscript.Field[float], energy: gtscript.Field[float]):
            with computation(PARALLEL), interval(...):
                energy = temperature_to_energy(temperature=temperature, mass=mass)
            

        @gtscript.function
        def chunk_temperature(water_energy: gtscript.Field[float], 
            water_mass: gtscript.Field[float],
            air_energy: gtscript.Field[float], 
            air_mass: gtscript.Field[float], 
            land_energy: gtscript.Field[float], 
            land_mass: gtscript.Field[float]) -> float:
            temp = 0.0
            if water_mass[0, 0, 0] != 0:
                temp += water_energy[0, 0, 0] / (constants.WATER_HEAT_CAPACITY * water_mass[0, 0, 0])
            if air_mass[0, 0, 0] != 0:
                temp += air_energy[0, 0, 0] / (constants.AIR_HEAT_CAPACITY * air_mass[0, 0, 0])
            if land_mass[0, 0, 0] != 0:
                temp += land_energy[0, 0, 0] / (constants.LAND_HEAT_CAPACITY * land_mass[0, 0, 0])
            return temp

        def compute_chunk_temperature(water_energy: gtscript.Field[float], 
                                        water_mass: gtscript.Field[float], 
                                        air_energy: gtscript.Field[float], 
                                        air_mass: gtscript.Field[float], 
                                        land_energy: gtscript.Field[float], 
                                        land_mass: gtscript.Field[float],
                                        temperature: gtscript.Field[float]) -> float:
            with computation(PARALLEL), interval(...):
                temperature = chunk_temperature(water_energy=water_energy, water_mass=water_mass, air_energy=air_energy, air_mass=air_mass, land_energy=land_energy, land_mass=land_mass)

        def compute_chunk_mass(water_mass: gtscript.Field[float],
                                air_mass: gtscript.Field[float], 
                                land_mass: gtscript.Field[float],
                                chunk_mass: gtscript.Field[float]) -> float:
            with computation(PARALLEL), interval(...):
                chunk_mass = water_mass[0, 0, 0] + air_mass[0, 0, 0] + land_mass[0, 0, 0]


        
        def sum_vertical_values(in_field: gtscript.Field[float],
                           out_field: gtscript.Field[float]):
            """
            Sum all the values of the input field on K dimensions and put the result in the output field at [I, J, 0]
            :param in_field:
            :param out_field:
            :return:
            """
            with computation(BACKWARD), interval(...):
                out_field = in_field[0, 0, 0] # First copy the field
            with computation(BACKWARD), interval(0, -1):
                out_field += out_field[0, 0, 1] # Then add the next element to the previous one
        

        def add_energy(input_energy: float,
                       water_energy: gtscript.Field[float], 
                       water_mass: gtscript.Field[float], 
                       air_energy: gtscript.Field[float], 
                       air_mass: gtscript.Field[float], 
                       land_energy: gtscript.Field[float], 
                       land_mass: gtscript.Field[float],
                       number_of_chunks: int = len(self)):
            """
            Distribute energy on all the components of the planet uniformly
            :param input_energy:
            :return:
            """
            with computation(PARALLEL), interval(...):
                chunk_input_energy = input_energy / number_of_chunks
                chunk_mass = (water_mass[0, 0, 0] + air_mass[0, 0, 0] + land_mass[0, 0, 0])
                if water_mass[0, 0, 0] != 0:
                    water_energy[0, 0, 0] += chunk_input_energy * (water_mass[0, 0, 0]/chunk_mass)
                if air_mass[0, 0, 0] != 0:
                    air_energy[0, 0, 0] += chunk_input_energy * (air_mass[0, 0, 0]/chunk_mass)
                if land_mass[0, 0, 0] != 0:
                    land_energy[0, 0, 0] += chunk_input_energy * (land_mass[0, 0, 0]/chunk_mass)

        self._add_energy = gtscript.stencil(definition=add_energy, backend=self.backend)
        self._compute_chunk_mass = gtscript.stencil(definition=compute_chunk_mass, backend=self.backend)
        self._compute_chunk_temperature = gtscript.stencil(definition=compute_chunk_temperature, backend=self.backend)
        self._sum_vertical_values = gtscript.stencil(definition=sum_vertical_values, backend=self.backend)
        self._temperature_to_energy_field = gtscript.stencil(definition=temperature_to_energy_field, backend=self.backend)
        self._compute_heat_transfer_coefficient = gtscript.stencil(definition=compute_heat_transfer_coefficient, backend=self.backend)
        self._compute_chunk_composition = gtscript.stencil(definition=compute_chunk_composition, backend=self.backend)

    def sum_horizontal_values(self, field: gtscript.Field[float]):
        """
        Sum all the values of the input field on I dimensions and put the result in the output field at [0, J, K]
        :param in_field:
        :param out_field:
        :return:
        """
        sum = 0
        for i in range(len(field)):
            for j in range(len(field[i])):
                sum += field[i][j][0]
        return sum

    @property
    def average_temperature(self) -> float:
        self._compute_chunk_temperature(self.water_energy, self.water_mass, self.air_energy, self.air_mass, self.land_energy, self.land_mass, self.chunk_temp)
        temp_total_temperature = gt_storage.empty(self.shape, dtype=float, backend=self.backend)
        self._sum_vertical_values(self.chunk_temp, temp_total_temperature)
        self._average_temperature = self.sum_horizontal_values(temp_total_temperature) / len(self)
        print("Computing average temperature")
        return self._average_temperature

    @property
    def total_mass(self) -> float:
        self._compute_chunk_mass(self.water_mass, self.air_mass, self.land_mass, self.chunk_mass)
        temp_total_mass = gt_storage.empty(self.shape, dtype=float, backend=self.backend)
        self._sum_vertical_values(self.chunk_mass, temp_total_mass)
        self._total_mass = self.sum_horizontal_values(temp_total_mass)
        print("Computing total mass")
        return self._total_mass


    @property
    def composition(self):
        composition_mass_dict = dict()
        self._compute_chunk_mass(self.water_mass, self.air_mass, self.land_mass, self.chunk_mass) # If not already computed
        water_composition = gt_storage.empty(self.shape, dtype=float, backend=self.backend)
        air_composition = gt_storage.empty(self.shape, dtype=float, backend=self.backend)
        land_composition = gt_storage.empty(self.shape, dtype=float, backend=self.backend)
        self._compute_chunk_composition(self.water_mass, self.air_mass, self.land_mass, self.chunk_mass, water_composition, air_composition, land_composition)
        self._sum_vertical_values(water_composition, water_composition)
        self._sum_vertical_values(air_composition, air_composition)
        self._sum_vertical_values(land_composition, land_composition)
        composition_mass_dict["WATER"] = self.sum_horizontal_values(water_composition)/len(self)
        composition_mass_dict["AIR"] = self.sum_horizontal_values(air_composition)/len(self)
        composition_mass_dict["LAND"] = self.sum_horizontal_values(land_composition)/len(self)
        return composition_mass_dict
    

    @property
    def carbon_flux_to_ocean(self):
        """
        The amount of carbon absorbed at every TIME_DELTA by all the ocean
        :return:
        """
        return 100_000

    @property
    def land_carbon_decay(self):
        """
        The amount of carbon released at every TIME_DELTA due to all the biomass decaying
        :return:
        """
        return 330_000

    @property
    def biosphere_carbon_absorption(self):
        """
        The amount of carbon absorbed at every TIME_DELTA due to all the biomass growing
        :return:
        """
        return 300_000

    def __str__(self):
        res = f"Earth : \n" \
              f"- Mass {self.total_mass}\n" \
              f"- Average temperature: {self.average_temperature}\n" \
              f"- Composition: \n\t{f'{chr(10) + chr(9)} '.join(str(round(value * 100, 2)) + '% ' + key for key, value in self.composition.items())}"
        return res
    
    
    def energy_to_temperature(self, energy: float, mass: float, heat_capacity: float):
        """
        Set the temperature of the component by computing the energy from the mass and the temperature
        :param temperature:
        :return:
        """
        return energy / (mass * heat_capacity)



    def compute_total_energy(self): # TODO: Only for testing purposes
        return sum(elem.energy for elem in self.not_nones())

    def receive_radiation(self, energy: float):
        self._add_energy(input_energy=(energy * (1 - self.albedo)), 
                          water_energy=self.water_energy,
                          water_mass=self.water_mass, 
                          air_energy=self.air_energy,
                          air_mass=self.air_mass,
                          land_energy=self.land_energy, 
                          land_mass=self.land_mass)

    def fill_with_water(self):
        """
        Fill the earth with water
        :return:
        """
        self.water_mass = gt_storage.from_array(np.full(shape=self.shape, fill_value=1000), backend=self.backend)
        water_temp = gt_storage.from_array(np.random.uniform(290, 310, self.shape), backend=self.backend)
        self._temperature_to_energy_field(water_temp, self.water_mass, self.water_energy)
    


    
