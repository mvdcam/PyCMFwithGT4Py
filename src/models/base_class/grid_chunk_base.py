from collections.abc import Collection, Iterator
from typing import Optional

from models.physical_class.chunk_component import ChunkComponent


class GridChunkBase(list[ChunkComponent]):
    """
    First layer of the grid chunk model.
    This class takes care of the aggregation of the different Chunk Components.
    """
    COMPONENTS = "WATER", "AIR", "LAND"
    water_component: ChunkComponent = None
    air_component: ChunkComponent = None
    land_component: ChunkComponent = None

    def reindex(self):
        """
        Search the parent class for this object in it and set the index variable accordingly
        :return:
        """
        self.index = self.earth.index(self)

    def __init__(self, components: Collection[ChunkComponent], *, index: int = None, earth=None):
        super().__init__()
        self.earth = earth
        self.index = index
        if self.index is None and self.earth is not None and self in self.earth:
            # No index provided but parent provided ? Find the object in the parent
            self.reindex()
        self.neighbours = self.earth.neighbours(self.index) if self.earth else None

        # Add only the components that are not empty
        for i, component in enumerate(components):
            if not component.is_empty():
                # Link the component's reference to the chunk to self
                component.chunk = self
                # Link the water/air/land _component object variable to the component of the corresponding type
                self[component.type.upper()] = component

    def __len__(self) -> int:
        """
        :return: the size of the grid chunk, that is equal to the amount of different components it contain
        """
        return sum(self[x] is not None for x in GridChunkBase.COMPONENTS)

    def __iter__(self) -> Iterator[ChunkComponent]:
        return (x for x in (self.water_component, self.air_component, self.land_component) if x is not None)

    def __contains__(self, __x: object) -> bool:
        if isinstance(__x, ChunkComponent):
            return any(__x == self[x] for x in GridChunkBase.COMPONENTS if self[x] is not None)
        return super().__contains__(__x)

    def __setitem__(self, key: str, value: Optional[ChunkComponent]):
        if key.upper() not in GridChunkBase.COMPONENTS:
            raise NotImplementedError(f"Component {key} is not a valid component type")
        self.__setattr__(f"{key.lower()}_component", value)

    def __getitem__(self, item: str):
        return self.__getattribute__(f"{item.lower()}_component")

    def __eq__(self, other: object):
        return isinstance(other, GridChunkBase) and all(self[x] == other[x] for x in GridChunkBase.COMPONENTS)

    def __ne__(self, other: "GridChunkBase"):
        return not self.__eq__(other)
