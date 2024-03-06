from typing import TYPE_CHECKING

from views.widgets.clear_canvas_widget import ClearCanvas

if TYPE_CHECKING:
    from controller.ToolbarArea.toolbar_area_controller import ToolbarController
    from controller.main_controller import MainController


class ClearCanvasController:
    def __init__(self, parent_controller: "ToolbarController", main_controller: "MainController"):
        self.parent_controller = parent_controller
        self.main_controller = main_controller
        self.view = ClearCanvas(self)

    def button_pressed(self):
        self.main_controller.clear_pressed()
