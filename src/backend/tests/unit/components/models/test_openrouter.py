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

        mock_instance.bind.assert_called_once_with(
            response_format={"type": "json_schema", "json_schema": component.response_schema}
        )
        assert model == mock_bound

    def test_update_build_config_outputs(self, component_class):
        component = component_class()
        build_config = {
            "outputs": [
                {"name": "structured_output", "hidden": True},
                {"name": "structured_output_dataframe", "hidden": True},
            ]
        }

        updated = component.update_build_config(build_config, {"schema": {}}, "response_schema")
        for output in updated["outputs"]:
            assert output["hidden"] is False

        updated = component.update_build_config(build_config, None, "response_schema")
        for output in updated["outputs"]:
            assert output["hidden"] is True
