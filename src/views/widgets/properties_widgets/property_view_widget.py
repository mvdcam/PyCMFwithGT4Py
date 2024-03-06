import random
from typing import TYPE_CHECKING

import matplotlib.cm
from PyQt5 import QtGui
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtWidgets import *

from constants import CANVAS_SIZE

if TYPE_CHECKING:
    from controller.CanvasArea.canvas_area_controller import CanvasAreaController
from models.physical_class.chunk_component import ChunkComponent
from models.physical_class.grid_chunk import GridChunk
from other.utils import index_to_2D


class PropertyViewWidget(QWidget):
    CLEAR_COLOR = QtGui.QColor("black")
    heatmap = matplotlib.cm.get_cmap('coolwarm')

    def temperature_to_color(self, temperature: float) -> QtGui.QColor:
        ratio = (temperature - self.lowest_temperature_spinbox.value()) / (
                    self.highest_temperature_spinbox.value() - self.lowest_temperature_spinbox.value())
        if ratio < 0.5:
            return QColor(0, 0, int(255 * (1 - ratio)))
        else:
            return QColor(int(255 * (1 - ratio)), 0, 0)

    def __init__(self, controller: "CanvasAreaController"):
        self.controller = controller
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.canvas = QLabel(self)
        self.canvas.setPixmap(QPixmap.fromImage(QImage(*CANVAS_SIZE, QImage.Format_RGB32)))
        self.canvas.setFixedSize(*CANVAS_SIZE)
        self.canvas.mouseMoveEvent = self.mouse_moved
        self.canvas.setMouseTracking(True)

        lowest_temperature_layout = QHBoxLayout()
        text = QLabel("Lowest temperature (°C) for heatmap: ")
        lowest_temperature_layout.addWidget(text)

        self.lowest_temperature_spinbox = QSpinBox(self)
        self.lowest_temperature_spinbox.setRange(-273, 100000)
        self.lowest_temperature_spinbox.setValue(0)
        lowest_temperature_layout.addWidget(self.lowest_temperature_spinbox)

        highest_temperature_layout = QHBoxLayout()
        text = QLabel("Highest temperature (°C) for heatmap: ")
        highest_temperature_layout.addWidget(text)

        self.highest_temperature_spinbox = QSpinBox(self)
        self.highest_temperature_spinbox.setRange(-273, 100000)
        self.highest_temperature_spinbox.setValue(30)
        highest_temperature_layout.addWidget(self.highest_temperature_spinbox)

        self.layout().addLayout(lowest_temperature_layout)
        self.layout().addLayout(highest_temperature_layout)
        self.layout().addWidget(self.canvas)
        button = QPushButton("Update")
        button.clicked.connect(self.update_pixels)
        self.layout().addWidget(button)
        self.clear()

    def update_pixels(self) -> None:
        self.canvas.pixmap().fill(PropertyViewWidget.CLEAR_COLOR)
        if not self.controller.main_controller.model:
            return
        image = QImage(*CANVAS_SIZE, QImage.Format_RGB32)
        image.fill(PropertyViewWidget.CLEAR_COLOR)
        min_value = self.lowest_temperature_spinbox.value() + 273.15
        max_value = self.highest_temperature_spinbox.value() + 273.15
        for component in self.controller.main_controller.model.earth.not_nones():
            temp = component.temperature + 273.15
            x, y = index_to_2D(component.index, self.controller.main_controller.model.earth.shape)
            r, g, b, _ = PropertyViewWidget.heatmap((temp - min_value) / (max_value - min_value))
            image.setPixel(x, y, QColor(r * 255, g * 255, b * 255).rgb())
        self.canvas.setPixmap(QPixmap.fromImage(image))

    def mouse_moved(self, e: QtGui.QMouseEvent):
        if self.controller.main_controller.model.earth.get_component_at(e.x(), e.y()):
            component = self.controller.main_controller.model.get_component_at(e.x(), e.y())
            self.canvas.setToolTip(f"Temperature: {component.temperature} C")
        else:
            self.canvas.setToolTip("")

    def clear(self):
        self.canvas.pixmap().fill(PropertyViewWidget.CLEAR_COLOR)
        self.update()


if __name__ == '__main__':
    from controller.main_controller import MainController

    app = QApplication([])
    c = CanvasAreaController(MainController())
    earth = c.main_controller.model.earth
    for i in range(8000):
        # sets a random index of earth to a new grid chunk of random temperature
        index = random.randint(0, len(earth) - 1)
        earth[random.randint(0, len(earth))] = GridChunk([ChunkComponent(1000, random.randint(250, 350), "Water")],
                                                         volume=1, index=index)
    c.view.show()

    app.exec_()
