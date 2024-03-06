from typing import TYPE_CHECKING

from PyQt5 import QtWidgets

if TYPE_CHECKING:
    from controller.ToolbarArea.subcontrollers.clear_canvas_controller import ClearCanvasController


class ClearCanvas(QtWidgets.QPushButton):
    def __init__(self, controller: "ClearCanvasController"):
        self.controller = controller
        super().__init__("Reset")
        self.clicked.connect(self.controller.button_pressed)
