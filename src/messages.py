class MessageToProcess:
    icon: str = "fa.check"
    icon_color: str = "#00FF00"
    text: str = "No error"


class ErrorMessageToProcess(MessageToProcess):
    icon = "ei.error"
    icon_color = "#ff0033"
    text = "There is an unexpected error"


class NoComponentBrushSelected(ErrorMessageToProcess):
    icon = "fa.warning"
    icon_color = "#eed202"
    text = "You cannot paint without first selecting a component"


class CannotPaintNow(ErrorMessageToProcess):
    icon = "fa.warning"
    icon_color = "#eed202"
    text = "You cannot paint right now."


class Loading(MessageToProcess):
    pass
