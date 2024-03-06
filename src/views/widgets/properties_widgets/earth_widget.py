from typing import TYPE_CHECKING

from PyQt5 import QtWidgets, QtGui

from other.utils import FloatValidator, LabelledWidget
from views.widgets.spin_box_slider import SpinBoxSlider

if TYPE_CHECKING:
    from controller.PhysicalPropArea.subcontrollers.earth_controller import EarthController


class EarthWidget(QtWidgets.QPushButton):
    def __init__(self, controller: "EarthController"):
        self.controller = controller
        super().__init__("Earth properties")
        self.clicked.connect(self.controller.button_pressed)


class EarthPopupWidget(QtWidgets.QDialog):
    validator = FloatValidator()

    def __init__(self, controller: "EarthController"):
        self.controller = controller
        super().__init__()
        self.setLayout(QtWidgets.QVBoxLayout())

        self.bottom_layout = QtWidgets.QHBoxLayout()
        self.confirm = QtWidgets.QPushButton("Confirm")
        self.confirm.clicked.connect(self.controller.confirmed)
        self.bottom_layout.addWidget(self.confirm)

        self.cancel = QtWidgets.QPushButton("Cancel")
        self.cancel.clicked.connect(self.controller.cancelled)
        self.bottom_layout.addWidget(self.cancel)

        self.radius_value = LabelledWidget(QtWidgets.QLineEdit, "Radius [m]", vertical=False)
        self.radius_value.setToolTip("The radius of the planet in meters")
        self.radius_value.setText(str(self.controller.main_controller.get_earth_radius()))
        self.radius_value.textChanged.connect(self.verify_radius)
        self.radius_value.setValidator(self.validator)
        self.layout().addWidget(self.radius_value)

        self.albedo = LabelledWidget(SpinBoxSlider, "Albedo", vertical=False)
        self.albedo.setToolTip("The proportion of sun rays reflected")
        self.layout().addWidget(self.albedo)

        self.layout().addLayout(self.bottom_layout)

    def showEvent(self, a0: QtGui.QShowEvent):
        self.radius_value.setText(str(self.controller.main_controller.get_earth_radius()))
        self.albedo.set_value(self.controller.main_controller.get_earth_albedo() * 100)
        super().showEvent(a0)

    def verify_radius(self):
        """
        Verify that the lineEdit is a valid float, and disables the confirm button if not
        :return:
        """
        try:
            float(self.radius_value.text())
        except ValueError:
            self.confirm.setDisabled(True)
        else:
            self.confirm.setEnabled(True)
