from typing import Any

from langflow.custom import Component
from langflow.field_typing.range_spec import RangeSpec
from langflow.inputs import (
    DropdownInput,
    IntInput,
    MessageTextInput,
    MultilineInput,
)
from langflow.schema.message import Message # Changed import
from langflow.template.field.base import Output


class TextProcessorComponent(Component):
    display_name = "Text Processor"
    name = "TextProcessor"
    description = "Performs text operations like append, prepend, wrap, and joining multiple ordered text inputs."
    icon = "pilcrow"  # Or "text-cursor-input", "combine"
    beta = True

    inputs = [
        DropdownInput(
            name="operation_mode",
            display_name="Operation Mode",
            options=["Append", "Prepend", "Wrap", "Join"],
            value="Append",
            info="Select the text processing operation.",
            real_time_refresh=True,
        ),
        MessageTextInput(
            name="input_text",
            display_name="Input Text",
            info="The primary text to process.",
            show=True,  # Initial show state, will be updated by update_build_config
        ),
        MultilineInput(
            name="delimiter_string",
            display_name="Delimiter / Wrapper String",
            info="String to append, prepend, or use for wrapping.",
            show=True,  # Initial show state
        ),
        IntInput(
            name="number_of_join_inputs",
            display_name="Number of Texts to Join",
            value=2,
            info="Specify how many text inputs to join. Inputs are ordered and created dynamically.",
            range_spec=RangeSpec(min=1, max=20),  # Adjust max as needed
            real_time_refresh=True,
            show=False,  # Initially hidden
        ),
        # Dynamic join_text_N inputs are added via update_build_config
    ]

    outputs = [
        Output(
            display_name="Processed Text",
            name="processed_text",
            method="build_processed_text",
            types=["Message"], # Changed to Message
        ),
    ]

    def update_build_config(self, build_config: dict, field_value: Any, field_name: str | None = None) -> dict:
        # Get current operation_mode and number_of_join_inputs from build_config
        current_op_mode = build_config.get("operation_mode", {}).get("value", "Append")
        if field_name == "operation_mode":
            current_op_mode = field_value

        current_num_join_inputs = build_config.get("number_of_join_inputs", {}).get("value", 2)
        if field_name == "number_of_join_inputs" and field_value is not None:
            try:
                current_num_join_inputs = int(field_value)
            except ValueError:
                current_num_join_inputs = 2 # Fallback to default if conversion fails

        # --- Visibility for Append/Prepend/Wrap inputs ---
        show_simple_inputs = current_op_mode in ["Append", "Prepend", "Wrap"]
        if "input_text" in build_config:
            build_config["input_text"]["show"] = show_simple_inputs
        if "delimiter_string" in build_config:
            build_config["delimiter_string"]["show"] = show_simple_inputs

        # --- Visibility for Join mode specific input (number_of_join_inputs) ---
        show_join_config_input = current_op_mode == "Join"
        if "number_of_join_inputs" in build_config:
            build_config["number_of_join_inputs"]["show"] = show_join_config_input

        # --- Dynamically add/remove join_text_N inputs ---
        # Store existing values before removing keys
        existing_join_values = {}
        for key in list(build_config.keys()):
            if key.startswith("join_text_") and isinstance(build_config[key], dict):
                existing_join_values[key] = build_config[key].get("value", "")

        # First, remove any existing join_text_N inputs to refresh the list
        keys_to_remove = [k for k in list(build_config.keys()) if k.startswith("join_text_")]
        for key in keys_to_remove:
            del build_config[key]

        if current_op_mode == "Join":
            # Ensure current_num_join_inputs is within a sane range if necessary, e.g., max 20 as per RangeSpec
            # For this example, we'll trust the IntInput's RangeSpec to mostly handle it on the frontend,
            # but a backend check could be added.
            # Clamp the value to what RangeSpec implies, or a reasonable maximum.
            max_join_inputs = build_config.get("number_of_join_inputs", {}).get("range_spec", {}).get("max", 20)
            min_join_inputs = build_config.get("number_of_join_inputs", {}).get("range_spec", {}).get("min", 1)
            
            current_num_join_inputs = max(min_join_inputs, min(current_num_join_inputs, max_join_inputs))


            for i in range(current_num_join_inputs):
                input_field_name = f"join_text_{i}"
                input_display_name = f"Text to Join {i + 1}"
                
                # Use stored value if available, otherwise default to empty string
                preserved_value = existing_join_values.get(input_field_name, "")

                join_input_field = MultilineInput( # Changed to MultilineInput
                    name=input_field_name,
                    display_name=input_display_name,
                    info=f"Text input {i + 1} for joining.",
                    value=preserved_value, # Use preserved value
                    show=True
                ).model_dump(by_alias=True, exclude_none=True)
                
                build_config[input_field_name] = join_input_field
            
        return build_config

    def build_processed_text(self) -> Message: # Changed return type
        mode = self.operation_mode
        result_text = ""

        # Handle input_text correctly if it's a Message object
        raw_input_text = getattr(self, "input_text", "")
        if isinstance(raw_input_text, Message):
            input_text_val = raw_input_text.text
        else:
            input_text_val = str(raw_input_text)
        
        delimiter_string_val = str(getattr(self, "delimiter_string", ""))

        if mode == "Append":
            result_text = f"{input_text_val}{delimiter_string_val}"
        elif mode == "Prepend":
            result_text = f"{delimiter_string_val}{input_text_val}"
        elif mode == "Wrap":
            result_text = f"{delimiter_string_val}{input_text_val}{delimiter_string_val}"
        elif mode == "Join":
            joined_parts = []
            # Ensure number_of_join_inputs is an int, defaulting if not set or invalid
            try:
                num_inputs = int(self.number_of_join_inputs)
            except (ValueError, TypeError):
                num_inputs = 0 # Or a default like 2, but 0 makes sense if it's not properly configured
            
            for i in range(num_inputs):
                text_part_raw = getattr(self, f"join_text_{i}", "")
                if isinstance(text_part_raw, Message):
                    joined_parts.append(text_part_raw.text or "") # Ensure text is not None
                else:
                    joined_parts.append(str(text_part_raw))
            result_text = "".join(joined_parts)
        
        self.status = f"Processed text in {mode} mode."
        return Message(text=result_text) # Changed to return Message