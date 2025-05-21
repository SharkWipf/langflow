import re
import unicodedata
from collections.abc import Callable
from concurrent import futures
from pathlib import Path

import chardet
import orjson
import yaml
from defusedxml import ElementTree

from langflow.schema import Data

# Types of files that can be read simply by file.read()
# and have 100% to be completely readable
TEXT_FILE_TYPES = [
    "txt",
    "md",
    "mdx",
    "csv",
    "json",
    "yaml",
    "yml",
    "xml",
    "html",
    "htm",
    "pdf",
    "docx",
    "py",
    "sh",
    "sql",
    "js",
    "ts",
    "tsx",
]

IMG_FILE_TYPES = ["jpg", "jpeg", "png", "bmp", "image"]


def normalize_text(text):
    return unicodedata.normalize("NFKD", text)


def is_hidden(path: Path) -> bool:
    return path.name.startswith(".")


def format_directory_path(path: str) -> str:
    """Format a directory path to ensure it's properly escaped and valid.

    Args:
    path (str): The input path string.

    Returns:
    str: A properly formatted path string.
    """
    return path.replace("\n", "\\n")


def split_base_relative_path(base_path: str, file_path: str) -> tuple[str, str]:
    """Return the resolved base path and the file path relative to it."""
    base = Path(base_path).resolve()
    file = Path(file_path).resolve()
    try:
        relative = file.relative_to(base)
    except ValueError:
        relative = file.name
    return str(base), str(relative)


# Ignoring FBT001 because the DirectoryComponent in 1.0.19
# calls this function without keyword arguments
def retrieve_file_paths(
    path: str,
    load_hidden: bool,  # noqa: FBT001
    recursive: bool,  # noqa: FBT001
    depth: int,
    types: list[str] = TEXT_FILE_TYPES,
    whitelist_regexes: list[str] | None = None,
    blacklist_regexes: list[str] | None = None,
) -> list[str]:
    path = format_directory_path(path)
    path_obj = Path(path)
    if not path_obj.exists() or not path_obj.is_dir():
        msg = f"Path {path} must exist and be a directory."
        raise ValueError(msg)

    def match_types(p: Path) -> bool:
        return any(p.suffix == f".{t}" for t in types) if types else True

    def match_whitelist(p: Path) -> bool:
        if not whitelist_regexes:
            return True
        return any(re.search(pattern, str(p)) for pattern in whitelist_regexes)

    def match_blacklist(p: Path) -> bool:
        if not blacklist_regexes:
            return True
        return not any(re.search(pattern, str(p)) for pattern in blacklist_regexes)

    def is_not_hidden(p: Path) -> bool:
        return not is_hidden(p) or load_hidden

    def walk_level(directory: Path, max_depth: int):
        directory = directory.resolve()
        prefix_length = len(directory.parts)
        for p in directory.rglob("*" if recursive else "[!.]*"):
            if len(p.parts) - prefix_length <= max_depth:
                yield p

    glob = "**/*" if recursive else "*"
    paths = walk_level(path_obj, depth) if depth else path_obj.glob(glob)
    return [
        str(p)
        for p in paths
        if p.is_file() and match_types(p) and is_not_hidden(p) and match_whitelist(p) and match_blacklist(p)
    ]


def partition_file_to_data(file_path: str, *, silent_errors: bool) -> Data | None:
    # Use the partition function to load the file
    from unstructured.partition.auto import partition

    try:
        elements = partition(file_path)
    except Exception as e:
        if not silent_errors:
            msg = f"Error loading file {file_path}: {e}"
            raise ValueError(msg) from e
        return None

    # Create a Data
    text = "\n\n".join([str(el) for el in elements])
    metadata = elements.metadata if hasattr(elements, "metadata") else {}
    metadata["file_path"] = file_path
    return Data(text=text, data=metadata)


def read_text_file(file_path: str) -> str:
    file_path_ = Path(file_path)
    raw_data = file_path_.read_bytes()
    result = chardet.detect(raw_data)
    encoding = result["encoding"]

    if encoding in {"Windows-1252", "Windows-1254", "MacRoman"}:
        encoding = "utf-8"

    return file_path_.read_text(encoding=encoding)


def read_docx_file(file_path: str) -> str:
    from docx import Document

    doc = Document(file_path)
    return "\n\n".join([p.text for p in doc.paragraphs])


def parse_pdf_to_text(file_path: str) -> str:
    from pypdf import PdfReader

    with Path(file_path).open("rb") as f:
        reader = PdfReader(f)
        return "\n\n".join([page.extract_text() for page in reader.pages])


def parse_text_file_to_data(
    file_path: str,
    *,
    silent_errors: bool,
    base_path: str | None = None,
) -> Data | None:
    try:
        if file_path.endswith(".pdf"):
            text = parse_pdf_to_text(file_path)
        elif file_path.endswith(".docx"):
            text = read_docx_file(file_path)
        else:
            text = read_text_file(file_path)

        # if file is json, yaml, or xml, we can parse it
        if file_path.endswith(".json"):
            text = orjson.loads(text)
            if isinstance(text, dict):
                text = {k: normalize_text(v) if isinstance(v, str) else v for k, v in text.items()}
            elif isinstance(text, list):
                text = [normalize_text(item) if isinstance(item, str) else item for item in text]
            text = orjson.dumps(text).decode("utf-8")

        elif file_path.endswith((".yaml", ".yml")):
            text = yaml.safe_load(text)
        elif file_path.endswith(".xml"):
            xml_element = ElementTree.fromstring(text)
            text = ElementTree.tostring(xml_element, encoding="unicode")
    except Exception as e:
        if not silent_errors:
            msg = f"Error loading file {file_path}: {e}"
            raise ValueError(msg) from e
        return None

    data = {"file_path": file_path, "text": text}
    if base_path:
        base, rel = split_base_relative_path(base_path, file_path)
        data["relative_path"] = rel
        data["base_path"] = base
    return Data(data=data)


# ! Removing unstructured dependency until
# ! 3.12 is supported
# def get_elements(
#     file_paths: List[str],
#     silent_errors: bool,
#     max_concurrency: int,
#     use_multithreading: bool,
# ) -> List[Optional[Data]]:
#     if use_multithreading:
#         data = parallel_load_data(file_paths, silent_errors, max_concurrency)
#     else:
#         data = [partition_file_to_data(file_path, silent_errors) for file_path in file_paths]
#     data = list(filter(None, data))
#     return data


def parallel_load_data(
    file_paths: list[str],
    *,
    silent_errors: bool,
    max_concurrency: int,
    load_function: Callable = parse_text_file_to_data,
) -> list[Data | None]:
    with futures.ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        loaded_files = executor.map(
            lambda file_path: load_function(file_path, silent_errors=silent_errors),
            file_paths,
        )
    # loaded_files is an iterator, so we need to convert it to a list
    return list(loaded_files)
