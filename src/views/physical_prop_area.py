from typing import TYPE_CHECKING

from PyQt5 import QtWidgets

if TYPE_CHECKING:
    from controller.PhysicalPropArea.physical_prop_area_controller import PhysicalPropAreaController


class PhysicalPropArea(QtWidgets.QWidget):
    def __init__(self, controller: "PhysicalPropAreaController"):
        self.controller = controller
        super().__init__()
        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().addWidget(self.controller.earth_controller.view)
        self.layout().addWidget(self.controller.sun_controller.view)
        self.layout().addWidget(self.controller.universe_controller.view)
