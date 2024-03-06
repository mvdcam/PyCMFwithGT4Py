from typing import TYPE_CHECKING

import qtawesome as qta

from constants import ICON_SIZE
from messages import *
from views.widgets.title_widget import Title

if TYPE_CHECKING:
    from controller.main_controller import MainController


class MessageController:
    message_stack: list[MessageToProcess] = []

    def __init__(self, parent_controller: "MainController"):
        self.parent_controller = parent_controller
        self.view = Title(controller=self)
        self.set_text_for_top_exception()

    def set_text(self, value: str):
        self.view.label.setText(value)

    def set_icon(self, name: str, color: str):
        self.view.icon.setPixmap(qta.icon(name, color=color).pixmap(*ICON_SIZE))

    def push_message(self, message: MessageToProcess):
        if len(self.message_stack) and isinstance(self.message_stack[-1], type(message)):
            return  # Don't push already existing exception
        self.message_stack.append(message)
        self.set_text_for_top_exception()

    def pop_message(self, message: type[MessageToProcess]):
        if not len(self.message_stack):
            return
        for i in range(len(self.message_stack) - 1, -1, -1):  # Look from the top of the stack
            if isinstance(self.message_stack[i], message):
                self.message_stack.pop(i)
        self.set_text_for_top_exception()

    def set_text_for_top_exception(self):
        message = self.message_stack[-1] if len(self.message_stack) else MessageToProcess
        self.set_icon(message.icon, message.icon_color)
        self.set_text(message.text)
        self.view.update()
