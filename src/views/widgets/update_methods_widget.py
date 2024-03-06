from typing import TYPE_CHECKING

from PyQt5 import QtWidgets, QtGui

if TYPE_CHECKING:
    from controller.update_methods_controller import UpdateMethodsController


class UpdateMethodsWidget(QtWidgets.QPushButton):
    def __init__(self, controller: "UpdateMethodsController"):
        self.controller = controller
        super().__init__("Update functions")
        self.clicked.connect(self.controller.button_pressed)


class UpdateMethodsPopupWidget(QtWidgets.QDialog):
    def __init__(self, controller: "UpdateMethodsController"):
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

        self.checkboxes = []

        for i, method in enumerate(self.controller.get_methods()):
            sub_layout = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(method.__name__)
            label.setToolTip(method.__doc__)
            label.setToolTipDuration(0)
            sub_layout.addWidget(label)

            checkbox = QtWidgets.QCheckBox()
            checkbox.setChecked(method.enabled)
            checkbox.setToolTip(method.__doc__)
            checkbox.setToolTipDuration(0)
            self.checkboxes.append(checkbox)
            sub_layout.addWidget(checkbox)
            self.layout().addLayout(sub_layout)

        self.layout().addLayout(self.bottom_layout)

    def showEvent(self, a0: QtGui.QShowEvent):
        for checkbox, method in zip(self.checkboxes, self.controller.get_methods()):
            checkbox.setChecked(method.enabled)
        super().showEvent(a0)
