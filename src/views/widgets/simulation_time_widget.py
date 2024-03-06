from typing import TYPE_CHECKING

import qtawesome as qta
from PyQt5 import QtWidgets

if TYPE_CHECKING:
    from controller.ToolbarArea.subcontrollers.simulation_time_controller import SimulationTimeController


class SimulationTimeWidget(QtWidgets.QWidget):
    def __init__(self, controller: "SimulationTimeController"):
        self.controller = controller
        super().__init__()
        self.setLayout(QtWidgets.QVBoxLayout())

        self.start_button = QtWidgets.QPushButton(qta.icon("fa.check", color="#228B22"), "Start Simulation")
        self.start_button.clicked.connect(self.controller.start_pressed)
        self.start_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.layout().addWidget(self.start_button)

        self.update_button = QtWidgets.QPushButton(qta.icon("mdi.step-forward", color="#228B22"), "Update Once")
        self.update_button.clicked.connect(self.controller.update_pressed)
        self.update_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.layout().addWidget(self.update_button)

        self.stop_button = QtWidgets.QPushButton(qta.icon("fa.hand-stop-o", color="#B22222"), "Stop Simulation")
        self.stop_button.clicked.connect(controller.stop_pressed)
        self.stop_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        self.pause_simulation = QtWidgets.QPushButton(qta.icon("fa5s.pause", color="#4169E1"), "Pause Simulation")
        self.pause_simulation.clicked.connect(controller.pause_pressed)
        self.pause_simulation.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.pause_simulation.setEnabled(False)
        self.pause_simulation.setMinimumWidth(150)
        self.layout().addWidget(self.pause_simulation)

        self.resume_simulation = QtWidgets.QPushButton(qta.icon("fa5s.play", color="#4169E1"), "Resume Simulation")
        self.resume_simulation.clicked.connect(controller.resume_pressed)
        self.resume_simulation.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.resume_simulation.setMinimumWidth(150)
