from dataclasses import dataclass
from typing import Union, Iterable, Iterator, TYPE_CHECKING

from PyQt5 import QtGui, QtWidgets
from PyQt5.QtGui import QValidator

from constants import COMPONENTS

if TYPE_CHECKING:
    from models.physical_class.grid_chunk import GridChunk


def color_from_chunk(chunk: "GridChunk"):
    return ComponentColor.from_chunk(chunk)


def index_to_2D(index: int, shape: tuple[int, int]) -> tuple[int, int]:
    """
    Convert the number from unidimensional to 2 dimensional
    :param index: the index to convert
    :param shape: Rows, Column
    :return:
    """
    return index % shape[1], index // shape[0]


class ComponentColor(QtGui.QColor):
    """
    Convert a component ratio to a color to draw
    """
    WATER = QtGui.QColor("blue")
    AIR = QtGui.QColor("white")
    LAND = QtGui.QColor("brown")
    DICT = dict()  # Saves the ratio used for that color

    def __init__(self, ratios: dict[str, float], *args, **kwargs):
        self.ratios = ratios
        super().__init__(*args, **kwargs)
        self.setRed(int(ComponentColor.WATER.red() * ratios.get("WATER", 0)) +
                    int(ComponentColor.AIR.red() * ratios.get("AIR", 0)) +
                    int(ComponentColor.LAND.red() * ratios.get("LAND", 0)))

        self.setGreen(int(ComponentColor.WATER.green() * ratios.get("WATER", 0)) +
                      int(ComponentColor.AIR.green() * ratios.get("AIR", 0)) +
                      int(ComponentColor.LAND.green() * ratios.get("LAND", 0)))

        self.setBlue(int(ComponentColor.WATER.blue() * ratios.get("WATER", 0)) +
                     int(ComponentColor.AIR.blue() * ratios.get("AIR", 0)) +
                     int(ComponentColor.LAND.blue() * ratios.get("LAND", 0)))
        ComponentColor.DICT[self.rgb()] = ratios

    @classmethod
    def from_chunk(cls, chunk: "GridChunk"):
        ratios = {component.type: chunk.get_ratio_of_component(component) for component in chunk}
        return cls(ratios)


class LabelledWidget(QtWidgets.QWidget):
    """
    Simple widget with a label to the left or on top of it
    """
    def __init__(self, widget_class: type[QtWidgets.QWidget], label: str, *, vertical: bool = True):
        super().__init__()
        self.widget_object = widget_class()
        self.setLayout(QtWidgets.QVBoxLayout() if vertical else QtWidgets.QHBoxLayout())
        self.layout().addWidget(QtWidgets.QLabel(label))
        self.layout().addWidget(self.widget_object)

    def __getattr__(self, item):
        """
        Redirects the attribute getting to the spinbox for abstraction
        :param item: the attribute to get (example the size, the value, the maximum value, etc ...)
        :return:
        """
        return self.widget_object.__getattribute__(item)


class FloatValidator(QValidator):
    """
    PyQt5 validator for float text edits to only accept numbers
    """
    def validate(self, string: str, pos: int) -> tuple['QValidator.State', str, int]:
        if not len(string):  # If the size is 0, it is a correct state
            return QValidator.Acceptable, string, pos
        if string[0] == ".":
            # If starts with a dot, verify that the rest is not invalid
            if self.validate(string[1:], pos)[0] == QValidator.Acceptable:
                return QValidator.Acceptable, string, pos
        if string.count("e") == 1:  # If there is one "e", check if before and after are valid
            if all(self.validate(x, 0)[0] == QValidator.Acceptable for x in string.split("e")):
                return QValidator.Acceptable, string, pos
        if string.count("-") == 1:  # If there is one "-", check if before and after are valid
            if all(self.validate(x, 0)[0] == QValidator.Acceptable for x in string.split("-")):
                return QValidator.Acceptable, string, pos
        if string.count("+") == 1:  # If there is one "-", check if before and after are valid
            if all(self.validate(x, 0)[0] == QValidator.Acceptable for x in string.split("+")):
                return QValidator.Acceptable, string, pos
        try:
            float(string)
        except ValueError:
            return QValidator.Invalid, string, pos
        else:
            return QValidator.Acceptable, string, pos
