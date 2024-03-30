from typing import Optional, Iterator

import numpy as np
import gt4py.cartesian.gtscript as gtscript
import gt4py.storage as gt_storage

from models.physical_class.grid_chunk import GridChunk


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

    def __setitem__(self, key, value: Optional[GridChunk]):
        """
        Makes sure that we match the number of active grind chunks when changing the Grid
        :param key:
        :param value:
        :return:
        """
        if self[key] is None and value is not None:
            self.nb_active_grid_chunks += 1
        elif self[key] is not None and value is None:
            self.nb_active_grid_chunks -= 1
        super().__setitem__(key, value)
        if value is not None:
            # If we insert an element, we need to recompute its neighbors
            value.neighbours = self.neighbours(value.index)
            for n in value.neighbours:
                n.neighbours = self.neighbours(n.index)


    def get_component_at(self, x, y=0, z=0):
        return self[x + y * self.shape[0] + z * (self.shape[0] + self.shape[1])]

    def set_component_at(self, component: GridChunk, x, y=0, z=0):
        self[x + y * self.shape[0] + z * (self.shape[0] + self.shape[1])] = component

    def neighbours(self, index: int) -> list[GridChunk]:
        """
        Yields the neighbouring element of the index, from front top left to back bottom right
        1D : [0,1,2,3,4,5]
        2D: [[0,1], [2,3], [4,5]
        3D: [[[0, 1], [2,3], [4,5]], [[6,7], [8,9], [10,11]]]
        :param index:
        :return:
        """
        res = []
        # 1D
        if len(self.shape) == 1:
            if index >= 1 and self[index - 1] is not None:  # Left
                res.append(self[index - 1])
            if index < len(self) - 1 and self[index + 1] is not None:  # Right
                res.append(self[index + 1])
        # 2D
        elif len(self.shape) == 2:
            if index >= self.shape[0] and self[index - self.shape[0]] is not None:  # Top
                res.append(self[index - self.shape[0]])
            if index % self.shape[0] != 0 and self[index - 1] is not None:  # Left
                res.append(self[index - 1])
            if (index + 1) % self.shape[0] != 0 and self[index + 1] is not None:  # Right
                res.append(self[index + 1])
            if index < self.shape[0] * self.shape[1] - self.shape[0] and self[index + self.shape[0]] is not None:  # Bot
                res.append(self[index + self.shape[0]])
        # 3D
        elif len(self.shape) == 3:
            # Front
            if 0 <= index - self.shape[0] * self.shape[1] and self[index - self.shape[0] * self.shape[1]] is not None:
                res.append(self[index - self.shape[0] * self.shape[1]])
            # Top
            if self.shape[0] <= index % (self.shape[0] * self.shape[1]) and self[index - self.shape[0]] is not None:
                res.append(self[index - self.shape[0]])
            # Left
            if 0 < index % self.shape[0] and self[index - 1] is not None:
                res.append(self[index - 1])
            # Right
            if 0 < (index + 1) % self.shape[0] and self[index + 1] is not None:
                res.append(self[index + 1])
            # Bottom
            if index % (self.shape[0] * self.shape[1]) < self.shape[0] * self.shape[1] - self.shape[0] and \
                    self[index + self.shape[0]] is not None:
                res.append(self[index + self.shape[0]])
            # Back
            if index + self.shape[0] * self.shape[1] < np.product(self.shape) and \
                    self[index + self.shape[0] * self.shape[1]] is not None:
                res.append(self[index + self.shape[0] * self.shape[1]])
        return res
