from unittest.mock import MagicMock

import pytest
from langflow.components.models import OpenRouterComponent

from tests.base import ComponentTestBaseWithoutClient


class TestOpenRouterComponent(ComponentTestBaseWithoutClient):
    @pytest.fixture
    def component_class(self):
        return OpenRouterComponent

    @pytest.fixture
    def default_kwargs(self):
        return {
            "api_key": "test-key",
            "provider": "OpenAI",
            "model_name": "openai/gpt-4o",
            "temperature": 0.7,
        }

    @pytest.fixture
    def file_names_mapping(self):
        return []

    def test_response_schema(self, component_class, mocker):
        component = component_class()
        component.api_key = "test-key"
        component.model_name = "openai/gpt-4o"
        component.response_schema = {
            "name": "test",
            "schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            "strict": True,
        }
        mock_instance = MagicMock()
        mock_bound = MagicMock()
        mock_instance.bind.return_value = mock_bound
        mocker.patch("langflow.components.models.openrouter.ChatOpenAI", return_value=mock_instance)

        model = component.build_model()

        mock_instance.bind.assert_called_once_with(response_format={"type": "json_schema", **component.response_schema})
        assert list(mock_instance.bind.call_args.kwargs["response_format"].keys()) == [
            "type",
            "name",
            "schema",
            "strict",
        ]
        assert model == mock_bound

    def test_structured_output(self, component_class, mocker):
        component = component_class()
        component.api_key = "test-key"
        component.model_name = "openai/gpt-4o"
        component.response_schema = {
            "name": "test",
            "schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            "strict": True,
        }
        mock_model = MagicMock()
        mock_model.bind.return_value = mock_model
        mocker.patch("langflow.components.models.openrouter.ChatOpenAI", return_value=mock_model)
        mocker.patch.object(component, "get_chat_result", return_value={"foo": "bar"})

        data = component.structured_output()

        assert data.data == {"results": {"foo": "bar"}}
