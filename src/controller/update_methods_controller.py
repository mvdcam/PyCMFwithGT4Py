from typing import TYPE_CHECKING

from views.widgets.update_methods_widget import UpdateMethodsWidget, UpdateMethodsPopupWidget

if TYPE_CHECKING:
    from controller.main_controller import MainController
    from controller.ToolbarArea.toolbar_area_controller import ToolbarController


class UpdateMethodsController:
    def __init__(self, parent_controller: "ToolbarController", main_controller: "MainController"):
        self.parent_controller = parent_controller
        self.main_controller = main_controller
        self.view = UpdateMethodsWidget(self)
        self.popup = UpdateMethodsPopupWidget(self)

    def get_methods(self):
        return self.main_controller.model.on_tick_methods

    def confirmed(self):
        self.popup.accept()
        self.popup.close()

    def cancelled(self):
        self.popup.reject()
        self.popup.close()

    def button_pressed(self):
        result = self.popup.exec_()
        if result:
            for checkbox, method in zip(self.popup.checkboxes, self.get_methods()):
                method.enabled = checkbox.isChecked()
