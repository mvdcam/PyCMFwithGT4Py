from typing import TYPE_CHECKING

from views.widgets.properties_widgets.sun_widget import SunWidget, SunPopupWidget

if TYPE_CHECKING:
    from controller.main_controller import MainController


class SunController:
    def __init__(self, main_controller: "MainController"):
        self.main_controller = main_controller
        self.view = SunWidget(self)
        self.popup = SunPopupWidget(self)

    def confirmed(self):
        self.popup.accept()
        self.popup.close()

    def cancelled(self):
        self.popup.reject()
        self.popup.close()

    def button_pressed(self):
        result = self.popup.exec_()
        if result:
            self.main_controller.set_sun_energy_per_second(float(self.popup.output.text()))
            self.main_controller.set_sun_radius(float(self.popup.radius_value.text()))
