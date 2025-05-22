import pytest
from langflow.components.processing.text_operations import TextOperationsComponent
from langflow.schema.message import Message

from tests.base import ComponentTestBaseWithoutClient


class TestTextOperationsComponent(ComponentTestBaseWithoutClient):
    @pytest.fixture
    def component_class(self):
        return TextOperationsComponent

    @pytest.fixture
    def default_kwargs(self):
        return {"operation": "Join Texts", "texts": ["a", "b"], "delimiter": ","}

    @pytest.fixture
    def file_names_mapping(self):
        return []

    def test_join_texts(self):
        component = TextOperationsComponent(operation="Join Texts", texts=["Hello", "World"], delimiter=" ")
        result = component.perform_operation()
        assert isinstance(result, Message)
        assert result.text == "Hello World"

    def test_append(self):
        component = TextOperationsComponent(operation="Append", text="Hello", append_text=" World")
        result = component.perform_operation()
        assert result.text == "Hello World"

    def test_prepend(self):
        component = TextOperationsComponent(operation="Prepend", text="World", prepend_text="Hello ")
        result = component.perform_operation()
        assert result.text == "Hello World"

    def test_wrap(self):
        component = TextOperationsComponent(operation="Wrap", text="middle", prefix="<", suffix=">")
        result = component.perform_operation()
        assert result.text == "<middle>"
