from __future__ import annotations

from typing import Any

from langflow.custom import Component
from langflow.inputs import DropdownInput, IntInput, MessageTextInput, MultilineInput
from langflow.field_typing.range_spec import RangeSpec
from langflow.io import Output
from langflow.schema.message import Message
from langflow.schema.dotdict import dotdict
from langflow.utils.component_utils import add_fields


class DynamicTextInputsComponent(Component):
    """Base component to manage a dynamic list of text inputs."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dynamic_inputs: dict[str, MultilineInput] = {}
        # Initialize dynamic inputs based on n_inputs if defined
        count = getattr(self, "n_inputs", 1)
        self._build_dynamic_inputs(count)

    def _build_dynamic_inputs(self, count: int) -> None:
        for i in range(1, count + 1):
            name = f"text_{i}"
            if name not in self.dynamic_inputs:
                field = MultilineInput(
                    name=name,
                    display_name=f"Text {i}",
                    dynamic=True,
                )
                self.dynamic_inputs[name] = field
                setattr(self, name, "")

    def _clear_dynamic_inputs(self, build_config: dotdict) -> None:
        for name in list(self.dynamic_inputs):
            build_config.pop(name, None)
            if hasattr(self, name):
                delattr(self, name)
            self.dynamic_inputs.pop(name, None)

    def update_dynamic_inputs(self, build_config: dotdict, count: int) -> None:
        self._clear_dynamic_inputs(build_config)
        self._build_dynamic_inputs(count)
        add_fields(build_config, {name: inp.to_dict() for name, inp in self.dynamic_inputs.items()})


class TextOperationsComponent(DynamicTextInputsComponent):
    """Perform simple text operations on inputs."""

    display_name = "Text Operations"
    description = "Perform prepend, append, wrap or join operations on text."
    icon = "file-text"
    name = "TextOperations"

    OPERATION_CHOICES = ["Prepend", "Append", "Wrap", "Join"]

    inputs = [
        MessageTextInput(
            name="text",
            display_name="Text",
            info="Base text for the operation.",
        ),
        DropdownInput(
            name="operation",
            display_name="Operation",
            options=OPERATION_CHOICES,
            real_time_refresh=True,
        ),
        MultilineInput(
            name="value",
            display_name="Value",
            info="Text value used for the operation.",
            dynamic=True,
            show=False,
        ),
        IntInput(
            name="n_inputs",
            display_name="Number of Inputs",
            value=1,
            range_spec=RangeSpec(min=1, step_type="int"),
            real_time_refresh=True,
            show=False,
        ),
    ]

    outputs = [Output(display_name="Result", name="result", method="process")]

    def update_build_config(self, build_config: dotdict, field_value: Any, field_name: str | None = None) -> dotdict:
        if field_name == "operation":
            if field_value == "Join":
                build_config["value"]["show"] = False
                build_config["n_inputs"]["show"] = True
                count = build_config["n_inputs"].get("value", 1)
                self.update_dynamic_inputs(build_config, count)
            else:
                build_config["value"]["show"] = True
                build_config["n_inputs"]["show"] = False
                self._clear_dynamic_inputs(build_config)
        elif field_name == "n_inputs" and getattr(self, "operation", None) == "Join":
            self.update_dynamic_inputs(build_config, int(field_value))
        return build_config

    def process(self) -> Message:
        operation = self.operation
        if operation == "Prepend":
            return Message(text=f"{self.value}{self.text}")
        if operation == "Append":
            return Message(text=f"{self.text}{self.value}")
        if operation == "Wrap":
            return Message(text=f"{self.value}{self.text}{self.value}")
        if operation == "Join":
            count = getattr(self, "n_inputs", 1)
            texts = [getattr(self, f"text_{i}", "") for i in range(1, count + 1)]
            return Message(text="".join(texts))
        return Message(text=self.text)
