from __future__ import annotations

from typing import List

from langflow.custom import Component
from langflow.io import DataInput, DataFrameInput, Output
from langflow.schema import Data, DataFrame
from langflow.schema.message import Message


class FormatDirectoryDataComponent(Component):
    display_name = "Format Directory Data"
    description = (
        "Format Data or DataFrame from the Directory component into a single text block"
    )
    icon = "file-lines"
    name = "FormatDirectoryData"

    inputs = [
        DataInput(
            name="data",
            display_name="Data",
            info="Data or list of Data objects to format",
            is_list=True,
            show=True,
            dynamic=True,
        ),
        DataFrameInput(
            name="df",
            display_name="DataFrame",
            info="DataFrame created from Directory component",
            show=True,
            dynamic=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Text",
            name="text",
            method="format_data",
            info="Formatted text for all files",
        )
    ]

    def _collect_data(self) -> List[Data]:
        if self.data:
            return self.data if isinstance(self.data, list) else [self.data]
        if self.df is not None:
            dataframe = self.df
            if isinstance(dataframe, DataFrame):
                return dataframe.to_data_list()
            return DataFrame(dataframe).to_data_list()
        msg = "Either Data or DataFrame input must be provided"
        raise ValueError(msg)

    def format_data(self) -> Message:
        data_list = self._collect_data()
        formatted_parts = []
        for item in data_list:
            if not isinstance(item, Data):
                item = Data(data=item)
            path = item.data.get("relative_path") or item.data.get("file_path", "")
            marker = "\n---\n" if path.lower().endswith(".md") else "```"
            content = str(item.get_text() or "").lstrip("\n").rstrip("\n")
            block = f"{path}:\n{marker}{content}"
            if content:
                block += "\n"
            block += f"{marker}\n\n"
            formatted_parts.append(block)
        result = "".join(formatted_parts).rstrip("\n") + "\n"
        self.status = result
        return Message(text=result)
