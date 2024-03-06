from typing import TYPE_CHECKING

from messages import Loading
from views.widgets.canvas_widget import CanvasWidget

if TYPE_CHECKING:
    from controller.CanvasArea.canvas_area_controller import CanvasAreaController
    from controller.main_controller import MainController


class CanvasController:
    painting_enabled: bool = True
    last_painted_positions: list[tuple[int, int]] = []

    def __init__(self, parent_controller: "CanvasAreaController", main_controller: "MainController"):
        self.main_controller = main_controller
        self.parent_controller = parent_controller
        self.view = CanvasWidget(controller=self)

    def get_brush_width(self):
        return self.main_controller.get_brush_width()

    def clear_canvas(self):
        self.view.clear()

    def is_painting_enabled(self):
        return self.painting_enabled

    def set_painting_enabled(self, value: bool):
        self.painting_enabled = value

    def mouse_moved(self, x: int, y: int):
        if self.main_controller.model:
            component = self.main_controller.model.get_component_at(x, y)
            self.view.setToolTip(component.__str__() or f"Component at {x}, {y}")
        else:
            self.view.setToolTip("")

    def mouse_engaged(self):
        pass

    def mouse_released(self):
        self.main_controller.process_message(Loading())
        self.main_controller.components_painted(*self.last_painted_positions)
        self.last_painted_positions = []
        self.main_controller.finish_process_message(Loading)

        self.parent_controller.temperature_view.update_pixels()
