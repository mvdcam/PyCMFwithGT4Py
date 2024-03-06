from typing import TYPE_CHECKING

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QPushButton

if TYPE_CHECKING:
    from controller.CanvasArea.canvas_area_controller import CanvasAreaController
    from controller.main_controller import MainController


class CanvasArea(QtWidgets.QWidget):
    def __init__(self, controller: "CanvasAreaController", main_controller: "MainController"):
        self.controller = controller
        self.main_controller = main_controller
        super().__init__()
        self.setLayout(QtWidgets.QVBoxLayout())
        sub_layout = QtWidgets.QVBoxLayout()
        self.title = self.main_controller.message_controller.view
        sub_layout.addWidget(self.title)
        button = QPushButton("Temperature View")
        button.clicked.connect(self.controller.show_temperature_view)
        sub_layout.addWidget(button)
        self.layout().addLayout(sub_layout)

        sub_layout = QtWidgets.QHBoxLayout()
        self.canvas = self.controller.canvas_controller.view
        self.earth_info = self.controller.text_edit_controller.view
        sub_layout.addWidget(self.canvas)
        sub_layout.addWidget(self.earth_info)

        self.layout().addLayout(sub_layout)

    def clear_canvas(self):
        self.canvas.clear()

    def set_canvas_enabled(self, value: bool):
        self.canvas.setEnabled(value)
