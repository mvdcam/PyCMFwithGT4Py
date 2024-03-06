from typing import TYPE_CHECKING

from PyQt5 import QtWidgets

from models.physical_class.universe import Universe

if TYPE_CHECKING:
    from controller.main_controller import MainController


class MainView(QtWidgets.QWidget):
    model: Universe

    def __init__(self, controller: "MainController"):
        self.controller = controller
        super().__init__()
        self.setLayout(QtWidgets.QVBoxLayout())

        self.toolbar_area_view = self.controller.toolbar_controller.view
        self.layout().addWidget(self.toolbar_area_view)

        self.canvas_area_view = self.controller.canvas_controller.view
        self.layout().addWidget(self.canvas_area_view)
