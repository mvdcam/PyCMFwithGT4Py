from typing import TYPE_CHECKING

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

if TYPE_CHECKING:
    from controller.ToolbarArea.subcontrollers.SelectComponent.slider_controller import SelectComponentSliderController


class LabelledDoubleSpinBoxSlider(QtWidgets.QWidget):
    slider: QtWidgets.QSlider
    spinbox: QtWidgets.QSpinBox
    lock_checkbox: QtWidgets.QCheckBox
    composition_button: QtWidgets.QPushButton

    def __init__(self, label: str, controller: "SelectComponentSliderController"):
        self.controller = controller
        super().__init__()
        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().addWidget(QtWidgets.QLabel(label.capitalize()))

        self.horizontal_slider_spinbox_other_layout = QtWidgets.QHBoxLayout()

        # Construct the slider widget
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setTickInterval(10)
        self.slider.setSingleStep(1)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.valueChanged.connect(self.controller.slider_changed)
        self.horizontal_slider_spinbox_other_layout.addWidget(self.slider)

        # Construct the spinbox (number text with arrows)
        self.spinbox = QtWidgets.QSpinBox()
        self.spinbox.setMinimum(0)
        self.spinbox.setMaximum(100)
        self.spinbox.valueChanged.connect(self.controller.spinbox_changed)
        self.horizontal_slider_spinbox_other_layout.addWidget(self.spinbox)

        # Construct the lock check box
        self.lock_checkbox = QtWidgets.QCheckBox("Lock")
        self.lock_checkbox.stateChanged.connect(self.controller.lock_changed)
        self.horizontal_slider_spinbox_other_layout.addWidget(self.lock_checkbox)

        # Add the 3 components from left to right to the layout
        self.layout().addLayout(self.horizontal_slider_spinbox_other_layout)


class SelectComponentSlider(LabelledDoubleSpinBoxSlider):
    def __init__(self, label: str, controller: "SelectComponentSliderController"):
        super(SelectComponentSlider, self).__init__(label, controller)
        self.composition_button = QtWidgets.QPushButton("Composition")
        self.horizontal_slider_spinbox_other_layout.insertWidget(0, self.composition_button)
