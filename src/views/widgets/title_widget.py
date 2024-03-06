from typing import TYPE_CHECKING

from PyQt5 import QtWidgets

if TYPE_CHECKING:
    from controller.exception_controller import MessageController


class Title(QtWidgets.QWidget):
    def __init__(self, controller: "MessageController"):
        self.controller = controller
        super().__init__()
        self.setLayout(QtWidgets.QHBoxLayout())

        self.icon = QtWidgets.QLabel()
        self.icon.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.layout().addWidget(self.icon)

        self.label = QtWidgets.QLabel()
        self.layout().addWidget(self.label)
