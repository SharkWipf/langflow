import json
from collections import OrderedDict
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic.v1 import SecretStr

from langflow.base.models.model import LCModelComponent
from langflow.base.models.openai_constants import (
    OPENAI_MODEL_NAMES,
    OPENAI_REASONING_MODEL_NAMES,
)
from langflow.field_typing import LanguageModel
from langflow.field_typing.range_spec import RangeSpec
from langflow.inputs import (
    BoolInput,
    DictInput,
    DropdownInput,
    IntInput,
    NestedDictInput,
    SecretStrInput,
    SliderInput,
    StrInput,
)
from langflow.logging import logger
from langflow.schema.data import Data
from langflow.schema.dataframe import DataFrame
from langflow.template.field.base import Output


class OpenAIModelComponent(LCModelComponent):
    display_name = "OpenAI"
    description = "Generates text using OpenAI LLMs."
    icon = "OpenAI"
    name = "OpenAIModel"

    inputs = [
        *LCModelComponent._base_inputs,
        DropdownInput(
            name="model_name",
            display_name="Model Name",
            advanced=False,
            options=OPENAI_MODEL_NAMES + OPENAI_REASONING_MODEL_NAMES,
            value=OPENAI_MODEL_NAMES[1],
            combobox=True,
            real_time_refresh=True,
        ),
        DropdownInput(
            name="response_format",
            display_name="Response Format",
            options=["text", "json_object", "json_schema"],
            value="text",
            info="Controls the output format. 'text' for standard output. 'json_object' for generic JSON. 'json_schema' to enforce a specific JSON schema (requires Response Schema input).",
            real_time_refresh=True,
            advanced=False,
        ),
        NestedDictInput(
            name="response_schema",
            display_name="Response Schema",
            info="JSON schema for structured outputs. Used only when Response Format is 'json_schema'.",
            show=False,
            input_types=["dict", "Data"],
        ),
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            advanced=True,
            info="The maximum number of tokens to generate. Set to 0 for unlimited tokens.",
            range_spec=RangeSpec(min=0, max=128000),
        ),
        DictInput(
            name="model_kwargs",
            display_name="Model Kwargs",
            advanced=True,
            info="Additional keyword arguments to pass to the model.",
        ),
        BoolInput(
            name="json_mode", # Retained as per original file, user said it's unrelated
            display_name="JSON Mode (Legacy)",
            advanced=True,
            info="Legacy: If True, it will output JSON regardless of passing a schema. Prefer using 'Response Format'.",
        ),
        StrInput(
            name="openai_api_base",
            display_name="OpenAI API Base",
            advanced=True,
            info="The base URL of the OpenAI API. "
            "Defaults to https://api.openai.com/v1. "
            "You can change this to use other APIs like JinaChat, LocalAI and Prem.",
        ),
        SecretStrInput(
            name="api_key",
            display_name="OpenAI API Key",
            info="The OpenAI API Key to use for the OpenAI model.",
            advanced=False,
            value="OPENAI_API_KEY",
            required=True,
        ),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.1,
            range_spec=RangeSpec(min=0, max=1, step=0.01),
            show=True,
        ),
        IntInput(
            name="seed",
            display_name="Seed",
            info="The seed controls the reproducibility of the job.",
            advanced=True,
            value=1,
        ),
        IntInput(
            name="max_retries",
            display_name="Max Retries",
            info="The maximum number of retries to make when generating.",
            advanced=True,
            value=5,
        ),
        IntInput(
            name="timeout",
            display_name="Timeout",
            info="The timeout for requests to OpenAI completion API.",
            advanced=True,
            value=700,
        ),
    ]

    # Outputs are now dynamically managed by update_outputs
    # We still need the methods for when they are active.
    # The base LCModelComponent.outputs are inherited.

    def build_model(self) -> LanguageModel:  # type: ignore[type-var]
        parameters = {
            "api_key": SecretStr(self.api_key).get_secret_value() if self.api_key else None,
            "model_name": self.model_name,
            "max_tokens": self.max_tokens or None,
            "model_kwargs": self.model_kwargs or {},
            "base_url": self.openai_api_base or "https://api.openai.com/v1",
            "seed": self.seed,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "temperature": self.temperature if self.temperature is not None else 0.1,
        }

        logger.info(f"Model name: {self.model_name}")
        if self.model_name in OPENAI_REASONING_MODEL_NAMES:
            logger.info("Getting reasoning model parameters")
            parameters.pop("temperature")
            parameters.pop("seed")
        
        llm_output = ChatOpenAI(**parameters)

        # Determine response_format binding
        binding_args = None
        if hasattr(self, "response_format"):
            if self.response_format == "json_object":
                binding_args = {"response_format": {"type": "json_object"}}
            elif self.response_format == "json_schema":
                if hasattr(self, "response_schema") and self.response_schema and self.response_schema != {}:
                    binding_args = {"response_format": {"type": "json_schema", "json_schema": self.response_schema}}
                else:
                    # Fallback for json_schema mode if schema is missing/empty
                    binding_args = {"response_format": {"type": "json_object"}}
        
        # Legacy json_mode, only if no explicit json_object/json_schema from response_format
        if not binding_args and self.json_mode:
            binding_args = {"response_format": {"type": "json_object"}}

        if binding_args:
            llm_output = llm_output.bind(**binding_args)

        return llm_output

    def structured_output(self) -> Data:
        message = self.text_response()
        content = getattr(message, "content", str(message))
        try:
            parsed = json.loads(content, object_pairs_hook=OrderedDict)
        except (TypeError, ValueError) as err:
            parsed = getattr(message, "parsed", None)
            if parsed is None:
                error_msg = "Unable to parse structured output"
                raise ValueError(error_msg) from err
        return Data(text_key="results", data={"results": parsed})

    def as_dataframe(self) -> DataFrame:
        data = self.structured_output().data.get("results")
        if isinstance(data, list):
            return DataFrame(data)
        return DataFrame([data])

    def _get_exception_message(self, e: Exception):
        """Get a message from an OpenAI exception.

        Args:
            e (Exception): The exception to get the message from.

        Returns:
            str: The message from the exception.
        """
        try:
            from openai import BadRequestError
        except ImportError:
            return None
        if isinstance(e, BadRequestError):
            message = e.body.get("message")
            if message:
                return message
        return None

    def update_build_config(self, build_config: dict, field_value: Any, field_name: str | None = None) -> dict:
        # Preserve existing logic for temperature and seed based on model_name
        if field_name == "model_name":
            if field_value in OPENAI_REASONING_MODEL_NAMES:
                build_config["temperature"]["show"] = False
                build_config["seed"]["show"] = False
            elif field_value in OPENAI_MODEL_NAMES:
                build_config["temperature"]["show"] = True
                build_config["seed"]["show"] = True
        
        current_response_format = build_config.get("response_format", {}).get("value", "text")

        if field_name == "response_format":
            current_response_format = field_value
        
        # Backwards compatibility for initial load
        if field_name is None: # Initial build
            if build_config.get("response_schema", {}).get("value") and build_config.get("response_schema", {}).get("value") != {}:
                build_config["response_format"]["value"] = "json_schema"
                current_response_format = "json_schema"
            elif build_config.get("json_mode", {}).get("value") is True:
                 build_config["response_format"]["value"] = "json_object"
                 current_response_format = "json_object"

        # Show/hide response_schema based on response_format
        # response_schema is only relevant for "json_schema" mode
        build_config["response_schema"]["show"] = (current_response_format == "json_schema")
        
        return build_config

    def update_outputs(self, frontend_node: dict, field_name: str, field_value: Any) -> dict:
        current_response_format = frontend_node.get("template", {}).get("response_format", {}).get("value", "text")
        if field_name == "response_format": # If response_format itself is changing
            current_response_format = field_value
        elif field_name is None: # Initial build, check for backward compatibility
            if frontend_node.get("template", {}).get("response_schema", {}).get("value") and frontend_node.get("template", {}).get("response_schema", {}).get("value") != {}:
                current_response_format = "json_schema"
            elif frontend_node.get("template", {}).get("json_mode", {}).get("value") is True:
                current_response_format = "json_object"


        # Always include all outputs from the class definition (including base class outputs)
        base_output_defs = [output.model_dump() for output in self.outputs]

        if current_response_format == "json_schema": # Only add structured outputs for json_schema mode
            structured_output_def = Output(name="structured_output", display_name="Structured Output", method="structured_output").model_dump()
            dataframe_output_def = Output(name="structured_output_dataframe", display_name="DataFrame", method="as_dataframe").model_dump()
            frontend_node["outputs"] = base_output_defs + [structured_output_def, dataframe_output_def]
        else: # "text" or "json_object" mode
            frontend_node["outputs"] = base_output_defs
            
        return frontend_node
