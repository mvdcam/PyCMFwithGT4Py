from typing import Optional, Iterator

import numpy as np
import gt4py.cartesian.gtscript as gtscript
import gt4py.storage as gt_storage


class EarthBase():
    """
    First layer of the earth model.
    The Python implementation of the Earth class to deal with the dunder method, index access, etc...
    There should be no reference to the physical properties of the Earth since it is taken care of in the second layer.
    """
    water_energy: gtscript.Field[float]
    water_mass: gtscript.Field[float]
    air_energy: gtscript.Field[float]
    air_mass: gtscript.Field[float]
    land_energy: gtscript.Field[float]
    land_mass: gtscript.Field[float]
    chunk_mass: gtscript.Field[float]
    heat_transfer_coefficient: gtscript.Field[float]
    specific_heat_capacity: gtscript.Field[float]
    carbon_ppm: gtscript.Field[float]
    backend: str


    def __init__(self, shape: tuple, 
                 water_energy: np.ndarray[float] = None, 
                 water_mass: np.ndarray[float] = None, 
                 air_energy: np.ndarray[float] = None, 
                 air_mass: np.ndarray[float] = None, 
                 land_energy: np.ndarray[float] = None, 
                 land_mass: np.ndarray[float] = None,
                 backend: str = "numpy", 
                 parent=None):
        self.shape = shape
        self.parent = parent
        self._total_mass = 0
        self._average_temperature = 0

        if water_energy is None:
            water_energy = np.zeros(shape)
        if water_mass is None:
            water_mass = np.zeros(shape)
        if air_energy is None:
            air_energy = np.zeros(shape)
        if air_mass is None:
            air_mass = np.zeros(shape)
        if land_energy is None:
            land_energy = np.zeros(shape)
        if land_mass is None:
            land_mass = np.zeros(shape)

        self.water_energy = gt_storage.from_array(water_energy, backend=backend)
        self.water_mass = gt_storage.from_array(water_mass, backend=backend)
        self.air_energy = gt_storage.from_array(air_energy, backend=backend)
        self.air_mass = gt_storage.from_array(air_mass, backend=backend)
        self.land_energy = gt_storage.from_array(land_energy, backend=backend)
        self.land_mass = gt_storage.from_array(land_mass, backend=backend)
        self.chunk_mass = gt_storage.empty(self.shape, dtype=float, backend=backend)
        self.chunk_temp = gt_storage.empty(self.shape, dtype=float, backend=backend)
        self.heat_transfer_coefficient = gt_storage.empty(self.shape, dtype=float, backend=backend)
        self.specific_heat_capacity = gt_storage.empty(self.shape, dtype=float, backend=backend)
        self.carbon_ppm = gt_storage.empty(self.shape, dtype=float, backend=backend)
        self.backend = backend
        
        


    def __len__(self):
        """
        The size of the Grid is always the static size, not the number of elements inside of it
        :return:
        """
        return np.product(self.shape)
