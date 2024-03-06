from constants import COMPONENTS
from controller.ToolbarArea.subcontrollers.SelectComponent.slider_controller import SelectComponentSliderController
from views.widgets.select_component_widget import SelectComponentPopupView


class SelectComponentPopupController:

    def __init__(self):
        self.balancing = False
        self.sub_controllers = [SelectComponentSliderController(component, i, parent_controller=self) for
                                i, component in enumerate(COMPONENTS)]
        self.view = SelectComponentPopupView(controller=self)

    # Pressing confirm/cancel button
    def confirmed(self):
        self.view.accept()

    def cancelled(self):
        self.view.reject()

    def balance_sliders(self, index: int, new_value: int):
        if self.balancing or index is None:
            return
        self.balancing = True
        # Count the remaining value to balance
        total_to_balance = 100 - new_value - sum(x.get_ratio() for x in self.sub_controllers if x.is_locked())
        # Count the amount of sliders not locked
        count_not_locked_not_index = sum(not x.is_locked() and x.index != index for x in self.sub_controllers)
        for slider_controller in self.sub_controllers:
            if slider_controller.index != index and not slider_controller.is_locked():
                slider_controller.set_ratio(total_to_balance // count_not_locked_not_index)
        self.balancing = False

    def lock_changed(self):
        """
        Locks all "Lock" checkbox when only 2 sliders are left unlocked
        :return:
        """
        count = sum(not x.is_locked() for x in self.sub_controllers)
        if count == 2:
            for elem in self.sub_controllers:
                if not elem.is_locked():
                    elem.set_locking_enabled(False)
        else:
            for elem in self.sub_controllers:
                if not elem.is_locking_enabled():
                    elem.set_locking_enabled(True)

    def get_remaining_to_balance(self):
        return 100 - sum(x.get_ratio() for x in self.sub_controllers if x.is_locked())

    def get_temperature(self):
        return self.view.temperature_spinbox.value()

    def get_mass(self):
        return self.view.mass_spinbox.value()
