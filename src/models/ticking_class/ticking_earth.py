import math

from gt4py.cartesian import gtscript

from models.ABC.ticking_model import TickingModel
from models.physical_class.earth import Earth
from models.ticking_class.ticking_grid_chunk import TickingGridChunk


class TickingEarth(Earth, TickingModel):
    """
    Third layer of the Earth Model.
    Contains only and all the rules for the model update.

    /!\ Those methods for update must be marked with @TickingModel.on_tick(enabled=True)
    """
    def __init__(self, shape: tuple, radius: float = 6.3781e6, *, parent=None, backend="numpy"):
        Earth.__init__(self, shape, radius, parent=parent, backend=backend)
        TickingModel.__init__(self)

    def update(self):
        """
        Special reimplementation of update to update all the components of the earth as well.
        Returns
        -------

        """
        super().update()

    # @TickingModel.on_tick(enabled=True)
    # def average_temperature(self):
    #     """
    #     Balances the temperature of each point on the earth
    #     :return:
    #     """
    #     temperature_gradiant = {}
    #     # First sweep of finding the temperature difference
    #     for elem in self.not_nones():
    #         for neighbour in elem.neighbours:
    #             if neighbour.index < elem.index or math.isclose(neighbour.temperature - elem.temperature, 0):
    #                 continue  # Already computed the other way around
    #             temperature_gradiant[(elem.index, neighbour.index)] = neighbour.temperature - elem.temperature
    #     # Second sweep to apply the difference
    #     for elem in self.not_nones():
    #         for neighbour in elem.neighbours:
    #             if neighbour.index < elem.index or (elem.index, neighbour.index) not in temperature_gradiant:
    #                 continue  # Already computed the other way around
    #             energy_exchanged = temperature_gradiant[(elem.index,
    #                                                      neighbour.index)] * elem.heat_transfer_coefficient * self.get_universe().TIME_DELTA / elem.surface
    #             elem.add_energy(energy_exchanged * elem.specific_heat_capacity)
    #             neighbour.add_energy(-energy_exchanged * elem.specific_heat_capacity)
        

    @TickingModel.on_tick(enabled=False)
    def carbon_cycle(self):
        """
        Globally computes carbon flow to be applied to each grid chunk
        :return:
        """
        carbon = self.CARBON_EMISSIONS_PER_TIME_DELTA - self.carbon_flux_to_ocean + self.land_carbon_decay - self.biosphere_carbon_absorption
        for chunk in self.not_nones():
            chunk.carbon_ppm += carbon / self.nb_active_grid_chunks
