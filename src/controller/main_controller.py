import threading
from typing import Optional

from constants import CANVAS_SIZE
from controller.CanvasArea.canvas_area_controller import CanvasAreaController
from controller.ToolbarArea.toolbar_area_controller import ToolbarController
from controller.exception_controller import MessageController
from messages import MessageToProcess
from models.physical_class.universe import Universe
from models.ticking_class.ticking_earth import TickingEarth
from models.ticking_class.ticking_sun import TickingSun
from views.main_view import MainView


class MainController:
    model: Universe
    simulation_thread: Optional[threading.Thread] = None

    def __init__(self):
        self.model = Universe()
        self.model.earth = TickingEarth(shape=CANVAS_SIZE, parent=self.model)
        self.model.sun = TickingSun()
        self.model.discover_everything()
        self.message_controller = MessageController(parent_controller=self)
        self.toolbar_controller = ToolbarController(parent_controller=self)
        self.canvas_controller = CanvasAreaController(parent_controller=self)
        self.view = MainView(controller=self)
        self.canvas_controller.clear_canvas()

    def clear_pressed(self):
        self.canvas_controller.clear_canvas()
        self.model = Universe()
        self.model.earth = TickingEarth(shape=CANVAS_SIZE, parent=self.model)
        self.model.sun = TickingSun()

    def start_pressed(self):
        self.canvas_controller.set_canvas_enabled(False)
        self.__start_simulation()

    def update_pressed(self):
        self.simulation_thread = threading.Thread(target=self.model.update_all, args=())
        self.simulation_thread.start()

    def pause_pressed(self):
        self.canvas_controller.set_canvas_enabled(True)
        self.__pause_simulation()

    def resume_pressed(self):
        self.canvas_controller.set_canvas_enabled(False)
        self.__resume_simulation()

    def stop_pressed(self):
        self.canvas_controller.set_canvas_enabled(True)
        self.__stop_simulation()

    def __start_simulation(self):
        self.simulation_thread = threading.Thread(target=self.model.start_simulation, args=())
        self.simulation_thread.start()

    def __pause_simulation(self):
        self.model.pause_updating()

    def __resume_simulation(self):
        self.simulation_thread = threading.Thread(target=self.model.resume_updating, args=())
        self.simulation_thread.start()

    def __stop_simulation(self):
        self.model.stop_updating()
        self.simulation_thread = None

    def get_brush_width(self):
        return self.toolbar_controller.get_brush_width()

    def finish_process_message(self, message: type[MessageToProcess]):
        self.message_controller.pop_message(message)

    def process_message(self, message: MessageToProcess):
        self.message_controller.push_message(message)

    def is_message_processing(self, exception: type[MessageToProcess]):
        return any(isinstance(e, exception) for e in self.message_controller.message_stack)

    def components_painted(self, *positions: tuple[int, int]):
        for x, y in set(positions):
            chunk = self.toolbar_controller.select_component_controller.get_grid_chunk().deep_copy()
            chunk.index = x + y * self.model.earth.shape[1]
            chunk.earth = self.model.earth
            chunk.neighbours = chunk.earth.neighbours(chunk.index)
            self.model.earth.set_component_at(chunk, x, y)

    def get_ratios(self):
        return self.toolbar_controller.select_component_controller.get_ratios()

    def get_grid_chunk(self):
        return self.toolbar_controller.select_component_controller.get_grid_chunk()

    def set_sun_energy_per_second(self, energy_per_second: float):
        self.model.sun.energy_radiated_per_second = energy_per_second

    def set_earth_radiation_ratio(self, earth_radiation_ratio: float):
        self.model.sun.earth_radiation_ratio = earth_radiation_ratio

    def get_energy_per_second(self):
        return self.model.sun.energy_radiated_per_second

    def set_time_delta(self, time_delta):
        self.model.TIME_DELTA = time_delta

    def get_time_delta(self):
        return self.model.TIME_DELTA

    def get_earth_radius(self):
        return self.model.earth.radius

    def set_earth_radius(self, radius: float):
        self.model.earth.radius = radius

    def get_earth_albedo(self):
        return self.model.earth.albedo

    def set_earth_albedo(self, albedo: float):
        self.model.earth.albedo = albedo

    def get_sun_radius(self):
        return self.model.sun.radius

    def set_sun_radius(self, radius: float):
        self.model.sun.radius = radius
