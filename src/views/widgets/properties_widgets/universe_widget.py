from typing import TYPE_CHECKING

from PyQt5 import QtWidgets, QtGui

from other.utils import LabelledWidget

if TYPE_CHECKING:
    from controller.PhysicalPropArea.subcontrollers.universe_controller import UniverseController


class UniverseWidget(QtWidgets.QPushButton):
    def __init__(self, controller: "UniverseController"):
        self.controller = controller
        super().__init__("Universe properties")
        self.clicked.connect(self.controller.button_pressed)


class UniversePopupWidget(QtWidgets.QDialog):
    def __init__(self, controller: "UniverseController"):
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

        self.time_delta_spinbox = LabelledWidget(QtWidgets.QDoubleSpinBox, "Temporal resolution", vertical=False)
        self.time_delta_spinbox.setToolTip("How many seconds between each updates of the simulation.")
        self.time_delta_spinbox.setMaximum(3600)
        self.time_delta_spinbox.setSingleStep(0.1)
        self.time_delta_spinbox.setMinimum(0.001)
        self.time_delta_spinbox.setValue(self.controller.main_controller.get_time_delta())
        self.layout().addWidget(self.time_delta_spinbox)

        self.layout().addLayout(self.bottom_layout)

    def showEvent(self, a0: QtGui.QShowEvent):
        self.time_delta_spinbox.setValue(self.controller.main_controller.get_time_delta())
        super(UniversePopupWidget, self).showEvent(a0)
