from typing import TYPE_CHECKING

from PyQt5 import QtWidgets

if TYPE_CHECKING:
    from controller.ToolbarArea.toolbar_area_controller import ToolbarController


class ToolbarArea(QtWidgets.QWidget):
    def __init__(self, controller: "ToolbarController"):
        self.controller = controller
        super().__init__()
        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().addWidget(self.controller.select_component_controller.view)
        self.layout().addWidget(self.controller.simulation_time_controller.view)
        self.layout().addWidget(self.controller.physical_prop_controller.view)
        self.layout().addWidget(self.controller.update_methods_controller.view)
