import pytest
from langflow.components.processing import JoinTextsComponent
from langflow.schema.message import Message

from tests.base import ComponentTestBaseWithoutClient


class TestJoinTextsComponent(ComponentTestBaseWithoutClient):
    @pytest.fixture
    def component_class(self):
        return JoinTextsComponent

    @pytest.fixture
    def default_kwargs(self):
        return {
            "texts": [Message(text="a"), Message(text="b")],
            "delimiter": "\n",
        }

    @pytest.fixture
    def file_names_mapping(self):
        return []

    def test_join_texts_basic(self):
        component = JoinTextsComponent(texts=[Message(text="one"), Message(text="two")], delimiter="-")
        result = component.join_texts()
        assert result.text == "one-two"

    def test_join_texts_multiline_delimiter(self):
        component = JoinTextsComponent(
            texts=[Message(text="x"), Message(text="y"), Message(text="z")],
            delimiter="---\n",
        )
        result = component.join_texts()
        assert result.text == "x---\ny---\nz"
