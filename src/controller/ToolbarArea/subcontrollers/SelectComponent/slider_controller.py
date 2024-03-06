from typing import TYPE_CHECKING

from views.widgets.select_component_slider import LabelledDoubleSpinBoxSlider

if TYPE_CHECKING:
    from controller.ToolbarArea.subcontrollers.SelectComponent.popup_controller import SelectComponentPopupController


class SelectComponentSliderController:
    def __init__(self, component_type: str, index: int, parent_controller: "SelectComponentPopupController" = None):
        self.type = component_type
        self.parent_controller = parent_controller
        self.index = index
        self.view = LabelledDoubleSpinBoxSlider(component_type, self)
        self.maximum = 100

    def lock_changed(self):
        if self.view.lock_checkbox.isChecked():
            self.view.slider.setDisabled(True)
            self.view.spinbox.setDisabled(True)
        else:
            self.view.slider.setEnabled(True)
            self.view.spinbox.setEnabled(True)

    def slider_changed(self):
        if self.view.slider.value() > self.parent_controller.get_remaining_to_balance():
            self.view.slider.setValue(self.parent_controller.get_remaining_to_balance())
        else:
            self.view.spinbox.setValue(self.view.slider.value())

    def spinbox_changed(self):
        if self.view.spinbox.value() > self.parent_controller.get_remaining_to_balance():
            self.view.spinbox.setValue(self.parent_controller.get_remaining_to_balance())
        else:
            self.view.slider.setValue(self.view.spinbox.value())
            self.parent_controller.balance_sliders(self.index, self.get_ratio())

    def set_ratio(self, value: int):
        self.view.slider.setValue(value)

    def is_locked(self):
        return self.view.lock_checkbox.isChecked()

    def set_locking_enabled(self, value: bool):
        if not value:
            self.view.lock_checkbox.setChecked(False)
        self.view.lock_checkbox.setEnabled(value)

    def is_locking_enabled(self):
        return self.view.lock_checkbox.isEnabled()

    def get_ratio(self):
        return self.view.slider.value()
