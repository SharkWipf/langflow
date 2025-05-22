import pandas as pd
from langflow.custom import Component
from langflow.io import (
    BoolInput,
    DataFrameInput,
    DropdownInput,
    IntInput,
    MessageTextInput,
    Output,
    StrInput,
)
from langflow.schema import DataFrame


class DataFrameOperationsComponent(Component):
    display_name = "DataFrame Operations"
    description = "Perform various operations on a DataFrame."
    icon = "table"

    # Available operations
    OPERATION_CHOICES = [
        "Add Column",
        "Drop Column",
        "Filter",
        "Head",
        "Rename Column",
        "Replace Value",
        "Select Columns",
        "Sort",
        "Tail",
        "Append",
        "Merge",
    ]

    inputs = [
        DataFrameInput(
            name="df",
            display_name="DataFrame",
            info="The input DataFrame(s) to operate on.",
            is_list=True,
        ),
        DropdownInput(
            name="operation",
            display_name="Operation",
            options=OPERATION_CHOICES,
            info="Select the DataFrame operation to perform.",
            real_time_refresh=True,
        ),
        StrInput(
            name="column_name",
            display_name="Column Name",
            info="The column name to use for the operation.",
            dynamic=True,
            show=False,
        ),
        MessageTextInput(
            name="filter_value",
            display_name="Filter Value",
            info="The value to filter rows by.",
            dynamic=True,
            show=False,
        ),
        BoolInput(
            name="ascending",
            display_name="Sort Ascending",
            info="Whether to sort in ascending order.",
            dynamic=True,
            show=False,
            value=True,
        ),
        StrInput(
            name="new_column_name",
            display_name="New Column Name",
            info="The new column name when renaming or adding a column.",
            dynamic=True,
            show=False,
        ),
        MessageTextInput(
            name="new_column_value",
            display_name="New Column Value",
            info="The value to populate the new column with.",
            dynamic=True,
            show=False,
        ),
        StrInput(
            name="columns_to_select",
            display_name="Columns to Select",
            dynamic=True,
            is_list=True,
            show=False,
        ),
        IntInput(
            name="num_rows",
            display_name="Number of Rows",
            info="Number of rows to return (for head/tail).",
            dynamic=True,
            show=False,
            value=5,
        ),
        MessageTextInput(
            name="replace_value",
            display_name="Value to Replace",
            info="The value to replace in the column.",
            dynamic=True,
            show=False,
        ),
        MessageTextInput(
            name="replacement_value",
            display_name="Replacement Value",
            info="The value to replace with.",
            dynamic=True,
            show=False,
        ),
        StrInput(
            name="merge_on",
            display_name="Merge On Column",
            info="Column name to merge on when performing a merge operation.",
            dynamic=True,
            show=False,
        ),
    ]

    outputs = [
        Output(
            display_name="DataFrame",
            name="output",
            method="perform_operation",
            info="The resulting DataFrame after the operation.",
        )
    ]

    def update_build_config(self, build_config, field_value, field_name=None):
        # Hide all dynamic fields by default
        dynamic_fields = [
            "column_name",
            "filter_value",
            "ascending",
            "new_column_name",
            "new_column_value",
            "columns_to_select",
            "num_rows",
            "replace_value",
            "replacement_value",
            "merge_on",
        ]
        for field in dynamic_fields:
            build_config[field]["show"] = False

        # Show relevant fields based on the selected operation
        if field_name == "operation":
            build_config["df"]["is_list"] = field_value in {"Append", "Merge"}
            if field_value == "Filter":
                build_config["column_name"]["show"] = True
                build_config["filter_value"]["show"] = True
            elif field_value == "Sort":
                build_config["column_name"]["show"] = True
                build_config["ascending"]["show"] = True
            elif field_value == "Drop Column":
                build_config["column_name"]["show"] = True
            elif field_value == "Rename Column":
                build_config["column_name"]["show"] = True
                build_config["new_column_name"]["show"] = True
            elif field_value == "Add Column":
                build_config["new_column_name"]["show"] = True
                build_config["new_column_value"]["show"] = True
            elif field_value == "Select Columns":
                build_config["columns_to_select"]["show"] = True
            elif field_value in {"Head", "Tail"}:
                build_config["num_rows"]["show"] = True
            elif field_value == "Replace Value":
                build_config["column_name"]["show"] = True
                build_config["replace_value"]["show"] = True
                build_config["replacement_value"]["show"] = True
            elif field_value in {"Append", "Merge"}:
                if field_value == "Merge":
                    build_config["merge_on"]["show"] = True

        return build_config

    def perform_operation(self) -> DataFrame:
        operation = self.operation

        if operation in {"Append", "Merge"}:
            if operation == "Append":
                return self.append_dataframes()
            return self.merge_dataframes()

        df_copy = self._single_df_copy(operation)

        if operation == "Filter":
            return self.filter_rows_by_value(df_copy)
        if operation == "Sort":
            return self.sort_by_column(df_copy)
        if operation == "Drop Column":
            return self.drop_column(df_copy)
        if operation == "Rename Column":
            return self.rename_column(df_copy)
        if operation == "Add Column":
            return self.add_column(df_copy)
        if operation == "Select Columns":
            return self.select_columns(df_copy)
        if operation == "Head":
            return self.head(df_copy)
        if operation == "Tail":
            return self.tail(df_copy)
        if operation == "Replace Value":
            return self.replace_values(df_copy)
        msg = f"Unsupported operation: {operation}"

        raise ValueError(msg)

    # Existing methods
    def filter_rows_by_value(self, df: DataFrame) -> DataFrame:
        return DataFrame(df[df[self.column_name] == self.filter_value])

    def sort_by_column(self, df: DataFrame) -> DataFrame:
        return DataFrame(df.sort_values(by=self.column_name, ascending=self.ascending))

    def drop_column(self, df: DataFrame) -> DataFrame:
        return DataFrame(df.drop(columns=[self.column_name]))

    def rename_column(self, df: DataFrame) -> DataFrame:
        return DataFrame(df.rename(columns={self.column_name: self.new_column_name}))

    def add_column(self, df: DataFrame) -> DataFrame:
        df[self.new_column_name] = [self.new_column_value] * len(df)
        return DataFrame(df)

    def select_columns(self, df: DataFrame) -> DataFrame:
        columns = [col.strip() for col in self.columns_to_select]
        return DataFrame(df[columns])

    # New methods
    def head(self, df: DataFrame) -> DataFrame:
        return DataFrame(df.head(self.num_rows))

    def tail(self, df: DataFrame) -> DataFrame:
        return DataFrame(df.tail(self.num_rows))

    def replace_values(self, df: DataFrame) -> DataFrame:
        df[self.column_name] = df[self.column_name].replace(self.replace_value, self.replacement_value)
        return DataFrame(df)

    def _df_is_list(self) -> bool:
        return isinstance(self.df, list)

    def _single_df_copy(self, operation: str) -> DataFrame:
        if self._df_is_list():
            if len(self.df) != 1:
                msg = f"{operation} operation requires a single DataFrame"
                raise ValueError(msg)
            df = self.df[0]
        else:
            df = self.df
        return df.copy()

    def append_dataframes(self) -> DataFrame:
        df_list = self.df if self._df_is_list() else [self.df]
        if len(df_list) < 2:
            return DataFrame(df_list[0]) if df_list else DataFrame()
        dataframes = [pd.DataFrame(df) for df in df_list]
        appended = pd.concat(dataframes, ignore_index=True)
        return DataFrame(appended.reset_index(drop=True))

    def merge_dataframes(self) -> DataFrame:
        if not self.merge_on:
            msg = "Merge operation requires 'merge_on' column"
            raise ValueError(msg)
        df_list = self.df if self._df_is_list() else [self.df]
        if len(df_list) < 2:
            return DataFrame(df_list[0]) if df_list else DataFrame()
        merged = pd.DataFrame(df_list[0])
        for df in df_list[1:]:
            merged = merged.merge(pd.DataFrame(df), on=self.merge_on)
        return DataFrame(merged.reset_index(drop=True))
