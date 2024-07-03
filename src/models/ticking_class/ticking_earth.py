import math

from gt4py.cartesian import gtscript
from gt4py.cartesian.gtscript import PARALLEL, BACKWARD, computation, interval, horizontal, region, I, J, K, IJ, IJK, Field
import gt4py.storage as gt_storage

from models.ABC.ticking_model import TickingModel
from models.physical_class.earth import Earth


class TickingEarth(Earth, TickingModel):
    """
    Third layer of the Earth Model.
    Contains only and all the rules for the model update.

    /!\ Those methods for update must be marked with @TickingModel.on_tick(enabled=True)
    """
    def __init__(self, shape: tuple, radius: float = 6.3781e6, *, parent=None, backend="numpy"):
        Earth.__init__(self, shape, radius, parent=parent, backend=backend)
        TickingModel.__init__(self)
        self.time_delta = self.get_universe().TIME_DELTA
        self.evaporation_rate = self.get_universe().EVAPORATION_RATE

        @gtscript.function
        def temp_coefficient(heat_transfer_coefficient: gtscript.Field[float],
                             specific_heat_capacity: gtscript.Field[float]):
            return heat_transfer_coefficient[0,0,0] * specific_heat_capacity[0,0,0] * self.time_delta
        

        def compute_energy_transfer(in_field: gtscript.Field[float], energy: gtscript.Field[float], heat_transfer_coefficient: gtscript.Field[float], specific_heat_capacity: gtscript.Field[float]):
            """
            compute the energy transfer between the grid chunk and its neighbors
            :param grid_chunk:
            :return:
            """
            with computation(PARALLEL), interval(...):
                coeff = temp_coefficient(heat_transfer_coefficient, specific_heat_capacity)
                energy += (in_field[1, 0, 0] - in_field[0, 0, 0]) * coeff
                energy += (in_field[-1, 0, 0] - in_field[0, 0, 0]) * coeff
                energy += (in_field[0, 1, 0] - in_field[0, 0, 0]) * coeff
                energy += (in_field[0, -1, 0] - in_field[0, 0, 0]) * coeff
                energy += (in_field[0, 0, 1] - in_field[0, 0, 0]) * coeff
                energy += (in_field[0, 0, -1] - in_field[0, 0, 0]) * coeff


        def water_evaporation(water_mass: gtscript.Field[float], air_mass: gtscript.Field[float]):
            """
            Evaporate water from the water component of the grid chunk
            :param grid_chunk:
            :return:
            """
            with computation(PARALLEL), interval(...):
                evaporated_mass = self.evaporation_rate * self.time_delta * water_mass
                water_mass -= evaporated_mass
                air_mass += evaporated_mass

        def carbon_cycle(carbon_ppm: gtscript.Field[float], carbon_per_chunk: float):
            """
            Globally computes carbon flow to be applied to each grid chunk
            :return:
            """
            with computation(PARALLEL), interval(...):
                carbon_ppm += carbon_per_chunk


        self._water_evaporation = gtscript.stencil(definition=water_evaporation, backend=self.backend)
        self._compute_energy_transfer = gtscript.stencil(definition=compute_energy_transfer, backend=self.backend)
        self._carbon_cycle = gtscript.stencil(definition=carbon_cycle, backend=self.backend)

    def update(self):
        """
        Special reimplementation of update to update all the components of the earth as well.
        Returns
        -------

        """
        super().update()

    @TickingModel.on_tick(enabled=True)
    def update_temperature(self):
        """
        Update the temperature of each grid chunk
        :return:
        """
        self._compute_chunk_temperature(self.water_energy, self.water_mass, self.air_energy, self.air_mass, self.land_energy, self.land_mass, self.chunk_temp)
        temp_energy = gt_storage.zeros(self.shape, dtype=float, backend=self.backend)
        self._compute_energy_transfer(self.chunk_temp, temp_energy, self.heat_transfer_coefficient, self.specific_heat_capacity, origin=self.origin)
        self._add_energy(temp_energy, self.water_energy, self.water_mass, self.air_energy, self.air_mass, self.land_energy, self.land_mass)


    @TickingModel.on_tick(enabled=False)
    def water_evaporation(self):
        """
        Evaporate water from the water component of the grid chunk
        :return:
        """
        self._water_evaporation(self.water_mass, self.air_mass)

        

    @TickingModel.on_tick(enabled=False)
    def carbon_cycle(self):
        """
        Globally computes carbon flow to be applied to each grid chunk
        :return:
        """
        carbon_per_chunk = (self.CARBON_EMISSIONS_PER_TIME_DELTA - self.carbon_flux_to_ocean + self.land_carbon_decay - self.biosphere_carbon_absorption) / len(self)
        self._carbon_cycle(self.carbon_ppm, carbon_per_chunk)
