from typing import TYPE_CHECKING

from PyQt5 import QtWidgets, QtCore

if TYPE_CHECKING:
    from controller.CanvasArea.subcontrollers.text_edit_controller import TextEditController


class TextEdit(QtWidgets.QWidget):
    def __init__(self, controller: "TextEditController"):
        self.controller = controller
        super().__init__()
        self.setLayout(QtWidgets.QVBoxLayout())

        self.text_edit = QtWidgets.QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setText(controller.main_controller.model.__str__())
        self.layout().addWidget(self.text_edit)

        sub_layout = QtWidgets.QHBoxLayout()
        self.checkbox = QtWidgets.QCheckBox("Auto-Refresh")
        self.checkbox.setToolTip("Refresh the text automatically every 200ms")
        self.checkbox.stateChanged.connect(self.controller.checkbox_state_changed)
        sub_layout.addWidget(self.checkbox)

        self.button = QtWidgets.QPushButton("Refresh Info")
        self.button.clicked.connect(self.controller.refresh)
        sub_layout.addWidget(self.button)

        self.layout().addLayout(sub_layout)

        self.auto_update_timer = QtCore.QTimer()
        self.auto_update_timer.timeout.connect(self.controller.refresh)
