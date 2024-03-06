from typing import TYPE_CHECKING

from views.widgets.properties_widgets.universe_widget import UniverseWidget, UniversePopupWidget

if TYPE_CHECKING:
    from controller.main_controller import MainController


class UniverseController:
    time_delta: float = None

    def __init__(self, main_controller: "MainController"):
        self.main_controller = main_controller
        self.view = UniverseWidget(self)
        self.popup = UniversePopupWidget(self)

    def confirmed(self):
        self.popup.accept()
        self.popup.close()

    def cancelled(self):
        self.popup.reject()
        self.popup.close()

    def button_pressed(self):
        result = self.popup.exec_()
        if result:
            self.time_delta = self.popup.time_delta_spinbox.value()
            self.main_controller.set_time_delta(self.time_delta)
