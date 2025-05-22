from typing import Any

from langflow.custom import Component
from langflow.io import (
    DropdownInput,
    IntInput,
    MessageTextInput,
    MultilineInput,
    Output,
)
from langflow.schema.dotdict import dotdict
from langflow.schema.message import Message


class TextOperationsComponent(Component):
    """Perform simple text operations."""

    display_name = "Text Operations"
    description = "Append, prepend, wrap text, or concatenate multiple texts."
    icon = "align-left"
    name = "TextOperations"

    OPERATION_CHOICES = ["Append", "Prepend", "Wrap", "Concat"]
    MAX_TEXT_FIELDS = 10

    inputs = [
        MessageTextInput(
            name="text",
            display_name="Text",
            info="Base text input.",
            multiline=True,
        ),
        DropdownInput(
            name="operation",
            display_name="Operation",
            options=OPERATION_CHOICES,
            value="Append",
            real_time_refresh=True,
        ),
        MultilineInput(
            name="value",
            display_name="Value",
            info="String used for append, prepend, or wrap operations.",
            show=True,
        ),
        MultilineInput(
            name="delimiter",
            display_name="Delimiter",
            info="Delimiter used when concatenating texts.",
            value="\n",
            show=False,
        ),
        IntInput(
            name="number_of_texts",
            display_name="Number of Texts",
            info="Number of text fields to concatenate.",
            value=2,
            dynamic=True,
            show=False,
        ),
    ]

    outputs = [
        Output(display_name="Result", name="result", method="perform_operation"),
    ]

    def update_build_config(self, build_config: dotdict, field_value: Any, field_name: str | None = None) -> dotdict:
        if field_name == "operation":
            is_concat = field_value == "Concat"
            build_config["text"]["show"] = not is_concat
            build_config["value"]["show"] = field_value in {"Append", "Prepend", "Wrap"}
            build_config["delimiter"]["show"] = is_concat
            build_config["number_of_texts"]["show"] = is_concat

            if not is_concat:
                for key in list(build_config.keys()):
                    if key.startswith("text_"):
                        build_config.pop(key)
            else:
                number = int(build_config["number_of_texts"]["value"])
                build_config = self._adjust_text_fields(build_config, number)
        elif field_name == "number_of_texts":
            try:
                number = int(field_value)
            except ValueError:
                return build_config
            build_config = self._adjust_text_fields(build_config, number)
        return build_config

    def _adjust_text_fields(self, build_config: dotdict, number: int) -> dotdict:
        if number > self.MAX_TEXT_FIELDS:
            number = self.MAX_TEXT_FIELDS
            build_config["number_of_texts"]["value"] = number
        existing = {k: v for k, v in build_config.items() if k.startswith("text_")}
        # remove extra fields
        for key in list(existing.keys()):
            idx = int(key.split("_")[1])
            if idx > number:
                build_config.pop(key)
        # add needed fields
        for i in range(1, number + 1):
            key = f"text_{i}"
            if key not in build_config:
                field = MultilineInput(name=key, display_name=f"Text {i}", multiline=True)
                build_config[key] = field.to_dict()
        return build_config

    def perform_operation(self) -> Message:
        operation = self.operation
        if operation == "Append":
            result = f"{self.text}{self.value}"
        elif operation == "Prepend":
            result = f"{self.value}{self.text}"
        elif operation == "Wrap":
            result = f"{self.value}{self.text}{self.value}"
        elif operation == "Concat":
            count = getattr(self, "number_of_texts", 0)
            texts = []
            for i in range(1, int(count) + 1):
                attr = f"text_{i}"
                text_val = getattr(self, attr, "")
                texts.append(str(text_val))
            result = self.delimiter.join(texts)
        else:
            msg = f"Unsupported operation: {operation}"
            raise ValueError(msg)

        self.status = result
        return Message(text=result)
