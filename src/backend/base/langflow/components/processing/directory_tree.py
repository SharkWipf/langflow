from __future__ import annotations

from pathlib import Path
from typing import Any

from langflow.custom import Component
from langflow.io import DataFrameInput, DataInput, Output
from langflow.schema import Data, DataFrame
from langflow.schema.message import Message


class DirectoryTreeComponent(Component):
    """Generate a tree representation from Directory component output."""

    display_name = "Directory Tree"
    description = "Create a tree view from Data or DataFrame returned by the Directory component."
    icon = "file-tree"
    name = "DirectoryTree"

    inputs = [
        DataInput(
            name="data",
            display_name="Data",
            info="Data or list of Data objects from Directory component",
            is_list=True,
            show=True,
            dynamic=True,
        ),
        DataFrameInput(
            name="df",
            display_name="DataFrame",
            info="DataFrame from Directory component",
            show=True,
            dynamic=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Tree",
            name="tree",
            method="build_tree",
            info="Tree-like directory structure",
        )
    ]

    def _collect_data(self) -> list[Data]:
        if self.data:
            return self.data if isinstance(self.data, list) else [self.data]
        if self.df is not None:
            dataframe = self.df
            if isinstance(dataframe, DataFrame):
                return dataframe.to_data_list()
            return DataFrame(dataframe).to_data_list()
        msg = "Either Data or DataFrame input must be provided"
        raise ValueError(msg)

    def _build_tree_dict(self, paths: list[str]) -> dict[str, Any]:
        tree: dict[str, Any] = {}
        for file_path in sorted(paths):
            parts = Path(file_path).parts
            node = tree
            for part in parts:
                node = node.setdefault(part, {})
        return tree

    def _dict_to_lines(self, node: dict[str, Any], prefix: str = "") -> list[str]:
        lines = []
        keys = list(node.keys())
        for idx, key in enumerate(keys):
            connector = "└── " if idx == len(keys) - 1 else "├── "
            lines.append(prefix + connector + key)
            subprefix = prefix + ("    " if idx == len(keys) - 1 else "│   ")
            lines.extend(self._dict_to_lines(node[key], subprefix))
        return lines

    def build_tree(self) -> Message:
        data_list = self._collect_data()
        paths = [
            item.data.get("relative_path") or item.data.get("file_path", "")
            for item in data_list
            if isinstance(item, Data)
        ]
        tree_dict = self._build_tree_dict(paths)
        lines = self._dict_to_lines(tree_dict)
        tree_text = "\n".join(lines)
        self.status = tree_text
        return Message(text=tree_text)
