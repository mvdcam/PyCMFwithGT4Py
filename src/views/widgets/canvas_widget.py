from typing import TYPE_CHECKING

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt

from constants import CANVAS_SIZE
from messages import CannotPaintNow, NoComponentBrushSelected
from other.utils import color_from_chunk

if TYPE_CHECKING:
    from controller.CanvasArea.subcontrollers.canvas_controller import CanvasController


class CanvasWidget(QtWidgets.QLabel):
    CLEAR_COLOR = QtGui.QColor("black")

    def __init__(self, controller: "CanvasController"):
        self.controller = controller
        super().__init__()
        self.mouse_is_painting = False
        self.setPixmap(QtGui.QPixmap(*CANVAS_SIZE))
        self.setFixedSize(*CANVAS_SIZE)
        self.setMouseTracking(True)

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        self.controller.mouse_engaged()

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        self.controller.mouse_released()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if self.rect().contains(e.pos()):
            self.controller.mouse_moved(e.x(), e.y())
        # If we are not pressing (keyDown) the left mouse button, return
        if Qt.LeftButton != e.buttons():
            return
        # If we cannot paint right now, return
        if not self.controller.is_painting_enabled():
            self.controller.main_controller.process_message(CannotPaintNow())
            return
        # If there is no component selected, return
        chunk = self.controller.main_controller.get_grid_chunk()
        if chunk is None:
            self.controller.main_controller.process_message(NoComponentBrushSelected())
            return
        with QtGui.QPainter(self.pixmap()) as painter:
            pen = QtGui.QPen()
            brush_color = color_from_chunk(chunk)  # color_from_ratio(ratios)
            pen.setColor(brush_color)
            width = self.controller.get_brush_width()
            pen.setWidth(width)
            painter.setPen(pen)
            x, y = e.x(), e.y()
            painter.drawPoint(e.x(), e.y())
            if width == 1:
                self.controller.last_painted_positions.append((x, y))
            else:
                for i in range(max(0, x - width // 2), min(CANVAS_SIZE[0], x + width // 2)):
                    for j in range(max(0, y - width // 2), min(CANVAS_SIZE[1], y + width // 2)):
                        self.controller.last_painted_positions.append((i, j))
        self.update()

    def clear(self):
        self.pixmap().fill(CanvasWidget.CLEAR_COLOR)
        self.update()
