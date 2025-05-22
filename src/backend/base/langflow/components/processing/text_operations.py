from langflow.custom import Component
from langflow.io import DropdownInput, MessageTextInput, MultilineInput, Output
from langflow.schema.message import Message


class TextOperationsComponent(Component):
    display_name = "Text Operations"
    description = "Perform various operations on text."
    icon = "paragraph"
    name = "TextOperations"

    OPERATION_CHOICES = ["Join Texts", "Append", "Prepend", "Wrap"]

    inputs = [
        DropdownInput(
            name="operation",
            display_name="Operation",
            options=OPERATION_CHOICES,
            info="Select the text operation to perform.",
            real_time_refresh=True,
        ),
        MessageTextInput(
            name="text",
            display_name="Text",
            dynamic=True,
            show=False,
        ),
        MessageTextInput(
            name="texts",
            display_name="Texts",
            is_list=True,
            dynamic=True,
            show=False,
        ),
        MultilineInput(
            name="delimiter",
            display_name="Delimiter",
            info="Delimiter used to join texts.",
            value="\n",
            show=False,
        ),
        MultilineInput(
            name="append_text",
            display_name="Append Text",
            show=False,
        ),
        MultilineInput(
            name="prepend_text",
            display_name="Prepend Text",
            show=False,
        ),
        MultilineInput(
            name="prefix",
            display_name="Prefix",
            show=False,
        ),
        MultilineInput(
            name="suffix",
            display_name="Suffix",
            show=False,
        ),
    ]

    outputs = [
        Output(display_name="Text", name="result", method="perform_operation"),
    ]

    def update_build_config(self, build_config, field_value, field_name=None):
        dynamic_fields = [
            "text",
            "texts",
            "delimiter",
            "append_text",
            "prepend_text",
            "prefix",
            "suffix",
        ]
        for field in dynamic_fields:
            build_config[field]["show"] = False

        if field_name == "operation":
            if field_value == "Join Texts":
                build_config["texts"]["show"] = True
                build_config["delimiter"]["show"] = True
            elif field_value == "Append":
                build_config["text"]["show"] = True
                build_config["append_text"]["show"] = True
            elif field_value == "Prepend":
                build_config["text"]["show"] = True
                build_config["prepend_text"]["show"] = True
            elif field_value == "Wrap":
                build_config["text"]["show"] = True
                build_config["prefix"]["show"] = True
                build_config["suffix"]["show"] = True
        return build_config

    def perform_operation(self) -> Message:
        operation = self.operation
        if operation == "Join Texts":
            delimiter = self.delimiter
            texts = self.texts or []
            result = delimiter.join(str(t) for t in texts)
        elif operation == "Append":
            result = f"{self.text or ''}{self.append_text or ''}"
        elif operation == "Prepend":
            result = f"{self.prepend_text or ''}{self.text or ''}"
        elif operation == "Wrap":
            result = f"{self.prefix or ''}{self.text or ''}{self.suffix or ''}"
        else:
            msg = f"Unsupported operation: {operation}"
            raise ValueError(msg)

        self.status = result
        return Message(text=result)
