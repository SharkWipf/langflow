import json
from collections import OrderedDict, defaultdict
from typing import Any

import httpx
from langchain_openai import ChatOpenAI
from pydantic.v1 import SecretStr

from langflow.base.models.model import LCModelComponent
from langflow.field_typing import LanguageModel
from langflow.field_typing.range_spec import RangeSpec
from langflow.inputs import (
    BoolInput,
    DropdownInput,
    IntInput,
    NestedDictInput,
    SecretStrInput,
    SliderInput,
    StrInput,
)
from langflow.schema.data import Data
from langflow.schema.dataframe import DataFrame
from langflow.template.field.base import Output

_PROPERTY_THRESHOLD_1 = 10
_PROPERTY_THRESHOLD_2 = 100


def _count_schema_property_keys(schema: Any) -> int:
    """Recursively count keys under every ``properties`` object."""
    count = 0

    def _walk(obj: Any):
        nonlocal count
        if isinstance(obj, dict):
            props = obj.get("properties")
            if isinstance(props, dict):
                for v in props.values():
                    count += 1
                    _walk(v)
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(schema)
    return count


def _rename_schema_properties(schema: Any):
    """Return schema copy with keys prefixed and mapping new->old."""
    schema_copy = json.loads(json.dumps(schema))
    total = _count_schema_property_keys(schema_copy)
    digits = 1 if total < _PROPERTY_THRESHOLD_1 else 2 if total < _PROPERTY_THRESHOLD_2 else 3

    counter = 0
    mapping_new_to_old: dict[str, str] = {}

    def _walk(obj: Any):
        nonlocal counter
        if isinstance(obj, dict):
            props = obj.get("properties")
            if isinstance(props, dict):
                new_props = OrderedDict()
                local_old_to_new = {}
                for old_key, val in props.items():
                    new_key = f"{counter:0{digits}d}_{old_key}"
                    counter += 1
                    mapping_new_to_old[new_key] = old_key
                    local_old_to_new[old_key] = new_key
                    new_props[new_key] = _walk(val)
                obj["properties"] = new_props
                if isinstance(obj.get("required"), list):
                    obj["required"] = [local_old_to_new.get(k, k) for k in obj["required"]]
            for k, v in obj.items():
                if k != "properties":
                    obj[k] = _walk(v)
        elif isinstance(obj, list):
            return [_walk(i) for i in obj]
        return obj

    renamed = _walk(schema_copy)
    return renamed, mapping_new_to_old


def _revert_json_keys(data: Any, mapping_new_to_old: dict[str, str]):
    if isinstance(data, dict):
        reverted = OrderedDict()
        for k, v in data.items():
            reverted[mapping_new_to_old.get(k, k)] = _revert_json_keys(v, mapping_new_to_old)
        return reverted
    if isinstance(data, list):
        return [_revert_json_keys(i, mapping_new_to_old) for i in data]
    return data


class OpenRouterComponent(LCModelComponent):
    """OpenRouter API component for language models."""

    display_name = "OpenRouter"
    description = (
        "OpenRouter provides unified access to multiple AI models from different providers through a single API."
    )
    icon = "OpenRouter"

    schema_reorder_workaround: bool = False

    inputs = [
        *LCModelComponent._base_inputs,
        SecretStrInput(
            name="api_key", display_name="OpenRouter API Key", required=True, info="Your OpenRouter API key"
        ),
        StrInput(
            name="site_url",
            display_name="Site URL",
            info="Your site URL for OpenRouter rankings",
            advanced=True,
        ),
        StrInput(
            name="app_name",
            display_name="App Name",
            info="Your app name for OpenRouter rankings",
            advanced=True,
        ),
        DropdownInput(
            name="provider",
            display_name="Provider",
            info="The AI model provider",
            options=["Loading providers..."],
            value="Loading providers...",
            real_time_refresh=True,
            required=True,
        ),
        DropdownInput(
            name="model_name",
            display_name="Model",
            info="The model to use for chat completion",
            options=["Select a provider first"],
            value="Select a provider first",
            real_time_refresh=True,
            required=True,
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
            show=False, # Initially hidden, shown by update_build_config
            input_types=["dict", "Data"],
        ),
        BoolInput(
            name="schema_reorder_workaround",
            display_name="Schema Reorder Workaround",
            info="Rename schema properties to preserve ordering. Used when Response Format is 'json_schema' and a Response Schema is provided.",
            show=False, # Initially hidden, shown by update_build_config
        ),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.7,
            range_spec=RangeSpec(min=0, max=2, step=0.01),
            info="Controls randomness. Lower values are more deterministic, higher values are more creative.",
            advanced=True,
        ),
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            info="Maximum number of tokens to generate",
            advanced=True,
        ),
    ]

    # Outputs are now dynamically managed by update_outputs
    # Base LCModelComponent.outputs are inherited.

    _schema_mapping: dict[str, str] | None = None

    def fetch_models(self) -> dict[str, list]:
        """Fetch available models from OpenRouter API and organize them by provider."""
        url = "https://openrouter.ai/api/v1/models"

        try:
            with httpx.Client() as client:
                response = client.get(url)
                response.raise_for_status()

                models_data = response.json().get("data", [])
                provider_models = defaultdict(list)

                for model in models_data:
                    model_id = model.get("id", "")
                    if "/" in model_id:
                        provider = model_id.split("/")[0].title()
                        provider_models[provider].append(
                            {
                                "id": model_id,
                                "name": model.get("name", ""),
                                "description": model.get("description", ""),
                                "context_length": model.get("context_length", 0),
                            }
                        )

                return dict(provider_models)

        except httpx.HTTPError as e:
            self.log(f"Error fetching models: {e!s}")
            return {"Error": [{"id": "error", "name": f"Error fetching models: {e!s}"}]}

    def build_model(self) -> LanguageModel:
        """Build and return the OpenRouter language model."""
        model_not_selected = "Please select a model"
        api_key_required = "API key is required"

        if not self.model_name or self.model_name == "Select a provider first":
            raise ValueError(model_not_selected)

        if not self.api_key:
            raise ValueError(api_key_required)

        api_key = SecretStr(self.api_key).get_secret_value()

        # Build base configuration
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "openai_api_key": api_key,
            "openai_api_base": "https://openrouter.ai/api/v1",
            "temperature": self.temperature if self.temperature is not None else 0.7,
        }

        # Add optional parameters
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens

        headers = {}
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name

        if headers:
            kwargs["default_headers"] = headers

        try:
            output = ChatOpenAI(**kwargs)
        except (ValueError, httpx.HTTPError) as err:
            error_msg = f"Failed to build model: {err!s}"
            self.log(error_msg)
            raise ValueError(error_msg) from err

        llm_output = ChatOpenAI(**kwargs)

        binding_args = None
        self._schema_mapping = None # Reset schema mapping

        if hasattr(self, "response_format"):
            if self.response_format == "json_object":
                binding_args = {"response_format": {"type": "json_object"}}
            elif self.response_format == "json_schema":
                if hasattr(self, "response_schema") and self.response_schema and self.response_schema != {}:
                    schema_to_use = self.response_schema
                    if getattr(self, "schema_reorder_workaround", False):
                        schema_to_use, mapping = _rename_schema_properties(self.response_schema)
                        self._schema_mapping = mapping
                    binding_args = {"response_format": {"type": "json_schema", "json_schema": schema_to_use}}
                else:
                    # Fallback for json_schema mode if schema is missing/empty
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
        if (
            parsed is not None
            and getattr(self, "schema_reorder_workaround", False)
            and getattr(self, "_schema_mapping", None)
        ):
            parsed = _revert_json_keys(parsed, self._schema_mapping)
        return Data(text_key="results", data={"results": parsed})

    def as_dataframe(self) -> DataFrame:
        data = self.structured_output().data.get("results")
        if isinstance(data, list):
            return DataFrame(data)
        return DataFrame([data])

    def _get_exception_message(self, e: Exception) -> str | None:
        """Get a message from an OpenRouter exception.

        Args:
            e (Exception): The exception to get the message from.

        Returns:
            str | None: The message from the exception, or None if no specific message can be extracted.
        """
        try:
            from openai import BadRequestError

            if isinstance(e, BadRequestError):
                message = e.body.get("message")
                if message:
                    return message
        except ImportError:
            pass
        return None

    def update_build_config(self, build_config: dict, field_value: Any, field_name: str | None = None) -> dict:
        """Update build configuration based on field updates."""
        # Provider and model_name logic
        if field_name is None or field_name == "provider" or (field_name == "api_key" and field_value): # Also refresh on API key entry
            try:
                provider_models = self.fetch_models()
                current_provider_value = build_config.get("provider", {}).get("value")
                
                build_config["provider"]["options"] = sorted(list(provider_models.keys()))
                if not current_provider_value or current_provider_value not in provider_models:
                     if build_config["provider"]["options"]:
                        build_config["provider"]["value"] = build_config["provider"]["options"][0]
                        current_provider_value = build_config["provider"]["options"][0]
                     else: # No providers loaded
                        build_config["provider"]["value"] = "Loading providers..."
                        build_config["model_name"]["options"] = ["Select a provider first"]
                        build_config["model_name"]["value"] = "Select a provider first"

                if field_name == "provider": # If provider changed, update model list
                    current_provider_value = field_value

                if current_provider_value and current_provider_value in provider_models:
                    models = provider_models[current_provider_value]
                    build_config["model_name"]["options"] = [model["id"] for model in models]
                    if models:
                        build_config["model_name"]["value"] = models[0]["id"]
                    else:
                        build_config["model_name"]["options"] = ["No models for this provider"]
                        build_config["model_name"]["value"] = "No models for this provider"
                    
                    tooltips = {
                        model["id"]: (f"{model['name']}\nContext Length: {model['context_length']}\n{model['description']}")
                        for model in models
                    }
                    build_config["model_name"]["tooltips"] = tooltips

            except httpx.HTTPError as e:
                self.log(f"Error updating build config: {e!s}")
                build_config["provider"]["options"] = ["Error loading providers"]
                build_config["provider"]["value"] = "Error loading providers"
                build_config["model_name"]["options"] = ["Error loading models"]
                build_config["model_name"]["value"] = "Error loading models"

        # Response Format logic
        current_response_format = build_config.get("response_format", {}).get("value", "text")

        if field_name == "response_format":
            current_response_format = field_value
        
        # Backwards compatibility for initial load
        if field_name is None: # Initial build
            if build_config.get("response_schema", {}).get("value") and build_config.get("response_schema", {}).get("value") != {}:
                build_config["response_format"]["value"] = "json_schema"
                current_response_format = "json_schema"
            # No equivalent of json_mode for OpenRouter to default to json_object

        # Show/hide response_schema and schema_reorder_workaround
        show_response_schema_input = (current_response_format == "json_schema")
        build_config["response_schema"]["show"] = show_response_schema_input
        
        # schema_reorder_workaround is relevant if response_schema is shown (i.e., response_format is 'json_schema')
        show_reorder_workaround = show_response_schema_input
        build_config["schema_reorder_workaround"]["show"] = show_reorder_workaround
        
        return build_config

    def update_outputs(self, frontend_node: dict, field_name: str, field_value: Any) -> dict:
        current_response_format = frontend_node.get("template", {}).get("response_format", {}).get("value", "text")
        if field_name == "response_format": # If response_format itself is changing
            current_response_format = field_value
        elif field_name is None: # Initial build, check for backward compatibility
            if frontend_node.get("template", {}).get("response_schema", {}).get("value") and frontend_node.get("template", {}).get("response_schema", {}).get("value") != {}:
                current_response_format = "json_schema"
            # No json_mode to check for OpenRouter

        base_output_defs = [
            Output(display_name="Message", name="text_output", method="text_response").model_dump(),
            Output(display_name="Language Model", name="model_output", method="build_model").model_dump(),
        ]

        if current_response_format == "json_schema": # Only add structured outputs for json_schema mode
            structured_output_def = Output(name="structured_output", display_name="Structured Output", method="structured_output").model_dump()
            dataframe_output_def = Output(name="structured_output_dataframe", display_name="DataFrame", method="as_dataframe").model_dump()
            frontend_node["outputs"] = base_output_defs + [structured_output_def, dataframe_output_def]
        else: # "text" or "json_object" mode
            frontend_node["outputs"] = base_output_defs
            
        return frontend_node
