from typing import TYPE_CHECKING

from views.widgets.text_edit_widget import TextEdit

if TYPE_CHECKING:
    from controller.CanvasArea.canvas_area_controller import CanvasAreaController
    from controller.main_controller import MainController


class TextEditController:
    def __init__(self, parent_controller: "CanvasAreaController", main_controller: "MainController"):
        self.parent_controller = parent_controller
        self.main_controller = main_controller
        self.view = TextEdit(controller=self)

    def checkbox_state_changed(self):
        if self.view.checkbox.isChecked():
            self.view.button.setEnabled(False)
            self.view.auto_update_timer.start(200)
        else:
            self.view.button.setEnabled(True)
            self.view.auto_update_timer.stop()

    def refresh(self):
        self.view.text_edit.setText(self.main_controller.model.__str__())
