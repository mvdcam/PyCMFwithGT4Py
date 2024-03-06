from typing import TYPE_CHECKING

from controller.CanvasArea.subcontrollers.canvas_controller import CanvasController
from controller.CanvasArea.subcontrollers.text_edit_controller import TextEditController
from views.canvas_area import CanvasArea
from views.widgets.properties_widgets.property_view_widget import PropertyViewWidget

if TYPE_CHECKING:
    from controller.main_controller import MainController


class CanvasAreaController:
    def __init__(self, parent_controller: "MainController"):
        self.main_controller = parent_controller
        self.canvas_controller = CanvasController(parent_controller=self, main_controller=self.main_controller)
        self.text_edit_controller = TextEditController(parent_controller=self, main_controller=self.main_controller)
        self.view = CanvasArea(controller=self, main_controller=self.main_controller)
        self.temperature_view = PropertyViewWidget(controller=self)

    def show_temperature_view(self):
        self.temperature_view.show()

    def clear_canvas(self):
        self.canvas_controller.clear_canvas()
        self.temperature_view.clear()

    def set_canvas_enabled(self, value: bool):
        self.canvas_controller.view.setEnabled(value)

    def get_canvas_as_qimage(self):
        return self.view.canvas.pixmap().toImage()

    def update_temperature_canvas(self):
        self.temperature_view.update_pixels()
