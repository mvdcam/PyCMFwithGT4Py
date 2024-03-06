from typing import TYPE_CHECKING

from PyQt5 import QtWidgets

if TYPE_CHECKING:
    from controller.ToolbarArea.toolbar_area_controller import ToolbarController


class ToolbarWidget(QtWidgets.QWidget):
    def __init__(self, controller: "ToolbarController"):
        super().__init__()
        self.setLayout(QtWidgets.QHBoxLayout())
        self.select_component_widget = controller.select_component_controller.view
        self.layout().addWidget(self.select_component_widget)
        self.simulation_time_widget = controller.simulation_time_controller.view
        self.select_time_widget = None
