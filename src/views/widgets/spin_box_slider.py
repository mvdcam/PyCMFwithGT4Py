from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt


class SpinBoxSlider(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.slider_changed)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.spinbox = QtWidgets.QSpinBox()
        self.spinbox.valueChanged.connect(self.spinbox_changed)
        self.spinbox.setMinimum(0)
        self.spinbox.setMaximum(100)

        self.valueChanged = self.spinbox.valueChanged

        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().addWidget(self.slider)
        self.layout().addWidget(self.spinbox)

    def slider_changed(self):
        self.spinbox.setValue(self.slider.value())

    def spinbox_changed(self):
        self.slider.setValue(self.spinbox.value())

    def set_value(self, value: int):
        self.slider.setValue(value)

    def get_value(self) -> int:
        return self.slider.value()
