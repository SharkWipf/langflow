import pytest
from langflow.components.processing.text_operations import TextOperationsComponent
from langflow.schema.message import Message


class TestTextOperationsComponent:
    @pytest.mark.parametrize(
        ("operation", "expected"),
        [
            ("Append", "HelloWorld"),
            ("Prepend", "WorldHello"),
            ("Wrap", "WorldHelloWorld"),
        ],
    )
    def test_basic_operations(self, operation, expected):
        comp = TextOperationsComponent()
        comp.text = "Hello"
        comp.operation = operation
        comp.value = "World"
        result = comp.perform_operation()
        assert isinstance(result, Message)
        assert result.text == expected

    def test_concat_operation(self):
        comp = TextOperationsComponent()
        comp.operation = "Concat"
        comp.delimiter = ","
        comp.number_of_texts = 3
        comp.text_1 = "A"
        comp.text_2 = "B"
        comp.text_3 = "C"
        result = comp.perform_operation()
        assert result.text == "A,B,C"
