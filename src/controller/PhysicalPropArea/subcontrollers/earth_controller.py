from typing import TYPE_CHECKING

from views.widgets.properties_widgets.earth_widget import EarthWidget, EarthPopupWidget

if TYPE_CHECKING:
    from controller.main_controller import MainController


class EarthController:
    def __init__(self, main_controller: "MainController"):
        self.main_controller = main_controller
        self.view = EarthWidget(self)
        self.popup = EarthPopupWidget(self)

    def confirmed(self):
        self.popup.accept()
        self.popup.close()

    def cancelled(self):
        self.popup.reject()
        self.popup.close()

    def button_pressed(self):
        result = self.popup.exec_()
        if result:
            self.main_controller.set_earth_radius(float(self.popup.radius_value.text()))
            self.main_controller.set_earth_albedo(self.popup.albedo.get_value() / 100)
