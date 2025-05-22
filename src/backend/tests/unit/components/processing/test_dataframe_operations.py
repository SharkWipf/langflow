import pandas as pd
import pytest
from langflow.components.processing.dataframe_operations import DataFrameOperationsComponent
from langflow.schema import DataFrame


@pytest.fixture
def sample_dataframe():
    data = {"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1], "C": ["a", "b", "c", "d", "e"]}
    return pd.DataFrame(data)


@pytest.mark.parametrize(
    ("operation", "expected_columns", "expected_values"),
    [
        ("Add Column", ["A", "B", "C", "D"], [1, 5, "a", 10]),
        ("Drop Column", ["A", "C"], None),
        ("Filter", ["A", "B", "C"], [3, 3, "c"]),
        ("Sort", ["A", "B", "C"], [5, 1, "e"]),
        ("Rename Column", ["Z", "B", "C"], None),
        ("Select Columns", ["A", "C"], None),
        ("Head", ["A", "B", "C"], [1, 5, "a"]),
        ("Tail", ["A", "B", "C"], [5, 1, "e"]),
        ("Replace Value", ["A", "B", "C"], [1, 5, "z"]),
    ],
)
def test_operations(sample_dataframe, operation, expected_columns, expected_values):
    component = DataFrameOperationsComponent()
    component.df = sample_dataframe
    component.operation = operation

    if operation == "Add Column":
        component.new_column_name = "D"
        component.new_column_value = 10
    elif operation == "Drop Column":
        component.column_name = "B"
    elif operation == "Filter":
        component.column_name = "A"
        component.filter_value = 3
    elif operation == "Sort":
        component.column_name = "A"
        component.ascending = False
    elif operation == "Rename Column":
        component.column_name = "A"
        component.new_column_name = "Z"
    elif operation == "Select Columns":
        component.columns_to_select = ["A", "C"]
    elif operation in {"Head", "Tail"}:
        component.num_rows = 1
    elif operation == "Replace Value":
        component.column_name = "C"
        component.replace_value = "a"
        component.replacement_value = "z"

    result = component.perform_operation()

    assert list(result.columns) == expected_columns
    if expected_values is not None and isinstance(expected_values, list):
        assert list(result.iloc[0]) == expected_values


def test_empty_dataframe():
    component = DataFrameOperationsComponent()
    component.df = pd.DataFrame()
    component.operation = "Head"
    component.num_rows = 3
    result = component.perform_operation()
    assert result.empty


def test_non_existent_column():
    component = DataFrameOperationsComponent()
    component.df = pd.DataFrame({"A": [1, 2, 3]})
    component.operation = "Drop Column"
    component.column_name = "B"
    with pytest.raises(KeyError):
        component.perform_operation()


def test_invalid_operation():
    component = DataFrameOperationsComponent()
    component.df = pd.DataFrame({"A": [1, 2, 3]})
    component.operation = "Invalid Operation"
    with pytest.raises(ValueError, match="Unsupported operation: Invalid Operation"):
        component.perform_operation()


def test_append_operation():
    df1 = pd.DataFrame({"A": [1], "B": [1]})
    df2 = pd.DataFrame({"A": [2], "B": [2]})
    component = DataFrameOperationsComponent()
    component.df = [df1, df2]
    component.operation = "Append"
    result = component.perform_operation()
    assert isinstance(result, DataFrame)
    assert len(result) == 2


def test_merge_operation():
    df1 = pd.DataFrame({"id": [1], "A": [1]})
    df2 = pd.DataFrame({"id": [1], "B": [2]})
    component = DataFrameOperationsComponent()
    component.df = [df1, df2]
    component.operation = "Merge"
    component.merge_on = "id"
    result = component.perform_operation()
    assert isinstance(result, DataFrame)
    assert set(result.columns) == {"id", "A", "B"}
