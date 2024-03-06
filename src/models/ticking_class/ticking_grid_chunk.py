from collections.abc import Collection

import models.physical_class.universe as universe
from models.ABC.ticking_model import TickingModel
from models.physical_class.chunk_component import ChunkComponent
from models.physical_class.grid_chunk import GridChunk


class TickingGridChunk(TickingModel, GridChunk):
    """
    Third layer of the Grid chunk Model.
    Contains only and all the rules for the model update.

    /!\ Those methods for update must be marked with @TickingModel.on_tick(enabled=True)
    """
    def __init__(self, components: Collection[ChunkComponent], volume: float, *, carbon_ppm=0, index: int = None,
                 earth=None):
        GridChunk.__init__(self, components, volume, index=index, earth=earth, carbon_ppm=carbon_ppm)
        TickingModel.__init__(self)

    @TickingModel.on_tick(enabled=True)
    def water_evaporation(self):
        if self.water_component is None:
            return
        evaporated_mass = universe.Universe.EVAPORATION_RATE * universe.Universe.TIME_DELTA * self.water_component.mass
        self.water_component.mass -= evaporated_mass
        if self.air_component is None:
            self.air_component = ChunkComponent(evaporated_mass, self.temperature, "AIR")
        else:
            self.air_component.mass += evaporated_mass
