from pathlib import Path

from langflow.base.data.utils import TEXT_FILE_TYPES, parallel_load_data, parse_text_file_to_data, retrieve_file_paths
from langflow.custom import Component
from langflow.io import (
    BoolInput,
    IntInput,
    MessageTextInput,
    MultiselectInput,
    StrInput,
)
from langflow.schema import Data
from langflow.schema.dataframe import DataFrame
from langflow.template import Output


class DirectoryComponent(Component):
    display_name = "Directory"
    description = "Recursively load files from a directory."
    icon = "folder"
    name = "Directory"

    inputs = [
        MessageTextInput(
            name="path",
            display_name="Path",
            info="Path to the directory to load files from. Defaults to current directory ('.')",
            value=".",
            tool_mode=True,
        ),
        MessageTextInput(
            name="base_path",
            display_name="Base Path",
            info="Base directory path. If provided, the relative path will be appended to this path.",
            advanced=True,
        ),
        MessageTextInput(
            name="relative_path",
            display_name="Relative Path",
            info="Path relative to the base path to search in.",
            advanced=True,
        ),
        MultiselectInput(
            name="types",
            display_name="File Types",
            info="File types to load. Select one or more types or leave empty to load all supported types.",
            options=TEXT_FILE_TYPES,
            value=[],
            combobox=True,
        ),
        IntInput(
            name="depth",
            display_name="Depth",
            info="Depth to search for files.",
            value=0,
        ),
        IntInput(
            name="max_concurrency",
            display_name="Max Concurrency",
            advanced=True,
            info="Maximum concurrency for loading files.",
            value=2,
        ),
        BoolInput(
            name="load_hidden",
            display_name="Load Hidden",
            advanced=True,
            info="If true, hidden files will be loaded.",
        ),
        BoolInput(
            name="recursive",
            display_name="Recursive",
            advanced=True,
            info="If true, the search will be recursive.",
        ),
        BoolInput(
            name="silent_errors",
            display_name="Silent Errors",
            advanced=True,
            info="If true, errors will not raise an exception.",
        ),
        BoolInput(
            name="use_multithreading",
            display_name="Use Multithreading",
            advanced=True,
            info="If true, multithreading will be used.",
        ),
        StrInput(
            name="whitelist_filters",
            display_name="Whitelist Filters",
            info="Regex patterns (one per line) to include specific files.",
            advanced=True,
        ),
        StrInput(
            name="blacklist_filters",
            display_name="Blacklist Filters",
            info="Regex patterns (one per line) to exclude specific files.",
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Data", name="data", method="load_directory"),
        Output(display_name="DataFrame", name="dataframe", method="as_dataframe"),
    ]

    def load_directory(self) -> list[Data]:
        path = self.path
        base_path = self.base_path
        relative_path = self.relative_path
        types = self.types
        depth = self.depth
        max_concurrency = self.max_concurrency
        load_hidden = self.load_hidden
        recursive = self.recursive
        silent_errors = self.silent_errors
        use_multithreading = self.use_multithreading
        whitelist_filters = self.whitelist_filters
        blacklist_filters = self.blacklist_filters

        if base_path:
            resolved_base = self.resolve_path(base_path)
            resolved_path = (
                self.resolve_path(str(Path(resolved_base) / relative_path)) if relative_path else resolved_base
            )
        else:
            resolved_path = self.resolve_path(path)
            resolved_base = resolved_path

        # If no types are specified, use all supported types
        types = TEXT_FILE_TYPES if not types else list(dict.fromkeys(types))

        valid_types = types

        def parse_filters(filters: str | None) -> list[str]:
            if not filters:
                return []
            return [line.strip() for line in filters.splitlines() if line.strip()]

        file_paths = retrieve_file_paths(
            resolved_path,
            load_hidden=load_hidden,
            recursive=recursive,
            depth=depth,
            types=valid_types,
            whitelist_regexes=parse_filters(whitelist_filters),
            blacklist_regexes=parse_filters(blacklist_filters),
        )

        loaded_data = []
        if use_multithreading:
            loaded_data = parallel_load_data(
                file_paths,
                silent_errors=silent_errors,
                max_concurrency=max_concurrency,
                load_function=lambda fp, *, silent_errors: parse_text_file_to_data(
                    fp,
                    silent_errors=silent_errors,
                    base_path=resolved_base,
                ),
            )
        else:
            loaded_data = [
                parse_text_file_to_data(
                    file_path,
                    silent_errors=silent_errors,
                    base_path=resolved_base,
                )
                for file_path in file_paths
            ]

        valid_data = [x for x in loaded_data if x is not None and isinstance(x, Data)]
        self.status = valid_data
        return valid_data

    def as_dataframe(self) -> DataFrame:
        return DataFrame(self.load_directory())
