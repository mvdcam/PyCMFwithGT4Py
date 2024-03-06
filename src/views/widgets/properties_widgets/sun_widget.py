from typing import TYPE_CHECKING

from PyQt5 import QtWidgets, QtGui

from other.utils import FloatValidator, LabelledWidget

if TYPE_CHECKING:
    from controller.PhysicalPropArea.subcontrollers.sun_controller import SunController


class SunWidget(QtWidgets.QPushButton):
    def __init__(self, controller: "SunController"):
        self.controller = controller
        super().__init__("Sun properties")
        self.clicked.connect(self.controller.button_pressed)


class SunPopupWidget(QtWidgets.QDialog):
    validator = FloatValidator()

    def __init__(self, controller: "SunController"):
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

        self.output = LabelledWidget(QtWidgets.QLineEdit, "Energy output per second [W]", vertical=False)
        self.output.setToolTip("The total amount of Watt radiated by the sun in all directions every second.")
        self.output.setText(str(self.controller.main_controller.get_energy_per_second()))
        self.output.setValidator(self.validator)
        self.output.textChanged.connect(self.verify_output)
        self.layout().addWidget(self.output)

        self.radius_value = LabelledWidget(QtWidgets.QLineEdit, "Radius [m]", vertical=False)
        self.radius_value.setToolTip("The radius of the star in meters")
        self.radius_value.setText(str(self.controller.main_controller.get_sun_radius()))
        self.radius_value.textChanged.connect(self.verify_radius)
        self.radius_value.setValidator(self.validator)
        self.layout().addWidget(self.radius_value)

        self.layout().addLayout(self.bottom_layout)

    def showEvent(self, a0: QtGui.QShowEvent):
        self.output.setText(str(self.controller.main_controller.get_energy_per_second()))
        super(SunPopupWidget, self).showEvent(a0)

    def verify_output(self):
        """
        Verify that the lineEdit is a valid float, and disables the confirm button if not
        :return:
        """
        try:
            float(self.output.text())
        except ValueError:
            self.confirm.setDisabled(True)
        else:
            self.confirm.setEnabled(True)

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
