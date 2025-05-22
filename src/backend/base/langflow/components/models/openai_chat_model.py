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
from langflow.io import Output
from langflow.logging import logger
from langflow.schema.data import Data
from langflow.schema.dataframe import DataFrame


class OpenAIModelComponent(LCModelComponent):
    display_name = "OpenAI"
    description = "Generates text using OpenAI LLMs."
    icon = "OpenAI"
    name = "OpenAIModel"

    outputs = [
        *LCModelComponent.outputs,
        Output(
            display_name="Structured Output",
            name="structured_output",
            method="structured_output",
            hidden=True,
        ),
        Output(
            display_name="DataFrame",
            name="structured_output_dataframe",
            method="structured_output_dataframe",
            hidden=True,
        ),
    ]

    inputs = [
        *LCModelComponent._base_inputs,
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
            name="json_mode",
            display_name="JSON Mode",
            advanced=True,
            info="If True, it will output JSON regardless of passing a schema.",
        ),
        NestedDictInput(
            name="response_schema",
            display_name="Response Schema",
            advanced=True,
            input_types=["NestedDict"],
            info="JSON schema for structured outputs.",
        ),
        DropdownInput(
            name="model_name",
            display_name="Model Name",
            advanced=False,
            options=OPENAI_MODEL_NAMES + OPENAI_REASONING_MODEL_NAMES,
            value=OPENAI_MODEL_NAMES[1],
            combobox=True,
            real_time_refresh=True,
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
        output = ChatOpenAI(**parameters)
        if self.response_schema:
            output = output.bind(response_format={"type": "json_schema", **self.response_schema})
        elif self.json_mode:
            output = output.bind(response_format={"type": "json_object"})

        return output

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
        if field_name in {"base_url", "model_name", "api_key"} and field_value in OPENAI_REASONING_MODEL_NAMES:
            build_config["temperature"]["show"] = False
            build_config["seed"]["show"] = False
        if field_name in {"base_url", "model_name", "api_key"} and field_value in OPENAI_MODEL_NAMES:
            build_config["temperature"]["show"] = True
            build_config["seed"]["show"] = True
        schema = field_value if field_name == "response_schema" else self.response_schema
        has_schema = bool(schema)
        build_config["structured_output"]["hidden"] = not has_schema
        build_config["structured_output_dataframe"]["hidden"] = not has_schema
        return build_config

    def structured_output(self) -> Data:
        if not self.response_schema:
            return Data(data={})
        result = self.text_response()
        content = result.content if hasattr(result, "content") else result
        try:
            parsed = json.loads(content, object_pairs_hook=OrderedDict)
        except Exception as exc:
            msg = "Failed to parse structured output"
            raise ValueError(msg) from exc
        return Data(text_key="results", data={"results": parsed})

    def structured_output_dataframe(self) -> DataFrame:
        data = self.structured_output().data.get("results")
        if isinstance(data, list):
            return DataFrame(data=data)
        if data is None:
            return DataFrame()
        return DataFrame(data=[data])
