from typing import TYPE_CHECKING

from controller.PhysicalPropArea.subcontrollers.earth_controller import EarthController
from controller.PhysicalPropArea.subcontrollers.sun_controller import SunController
from controller.PhysicalPropArea.subcontrollers.universe_controller import UniverseController
from views.physical_prop_area import PhysicalPropArea

if TYPE_CHECKING:
    from controller.main_controller import MainController
    from controller.ToolbarArea.toolbar_area_controller import ToolbarController


class PhysicalPropAreaController:
    def __init__(self, parent_controller: "ToolbarController", main_controller: "MainController"):
        self.parent_controller = parent_controller
        self.main_controller = main_controller
        self.universe_controller = UniverseController(main_controller=main_controller)
        self.sun_controller = SunController(main_controller=main_controller)
        self.earth_controller = EarthController(main_controller=main_controller)
        self.view = PhysicalPropArea(self)
