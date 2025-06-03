from __future__ import annotations

import json
from typing import Any, Dict, List, Union

from jinja2 import BaseLoader, Environment, select_autoescape, TemplateNotFound
from langflow.custom import Component
from langflow.io import DataInput, DropdownInput, HandleInput, Output
from langflow.inputs import MultilineInput
from langflow.schema import Data, DataFrame
from langflow.schema.message import Message

# Jinja2 template as a string
COMPLETE_REPORT_TEMPLATE = """
{# =============================================================================
  complete_report.jinja2

  Prints out a sequence of (optional) fields in this order:
    1. Prelude   (printed at top / bottom / both, depending on prelude_position)
    2. Assignment (printed at top / bottom / both, depending on assignment_position)
    3. Intended design
    4. Intended usage
    5. 3rd party documentation
    6. 1st party documentation
    7. 1st party code
    8. 1st party misc files
    9. Logs
   10. Work already done
   11. Work to do
   12. Diff

  Rules:
    • Any field that is “undefined” or an empty string will be skipped.
    • Every field, once actually emitted, is preceded by a header block:
        ───────────────
        FieldName:
        ───────────────
      except the very *first* field in the entire output, which omits the
      initial “───” and blank lines, but still prints “FieldName:” and the
      following “───”.  In other words, “first field” header looks like:
        FieldName:
        ───────────────
      (no leading horizontal rule).
    • “Prelude” and “Assignment” each have a `*_position` (one of “top”, “bottom”,
      or “both”).  If you say `"both"`, then after printing it in the “top” region,
      when you come to print it again in the “bottom” region, you must label it
      “Repeat of Prelude:” or “Repeat of Assignment:”, instead of just
      “Prelude:” or “Assignment:”.
    • We keep track of:
        – `ns.first` (have we printed anything at all yet?)
        – `ns.prelude_printed` (have we already printed Prelude once?)
        – `ns.assignment_printed` (have we already printed Assignment once?)

  To render:
    render “complete_report.jinja2” with a context dictionary containing
    all of the above variables.  Any field that’s missing or empty is simply skipped.
# ============================================================================= #}
{# ---- initialize our “bookkeeping” namespace ---- #}
{% set ns = namespace(
      first=true,
      prelude_printed=false,
      assignment_printed=false
) %}
{# ==== 1) PRELUDE @ TOP if requested ==== #}
{% if prelude is defined and prelude %}
  {% if prelude_position in ['top', 'both'] %}
    {% set label = "Prelude" %}
    {% if ns.first %}
{{ label }}:
---

{{ prelude }}

      {% set ns.first = false %}
    {% else %}
---
{{ label }}:
---

{{ prelude }}

    {% endif %}
    {% set ns.prelude_printed = true %}
  {% endif %}
{% endif %}
{# ==== 2) ASSIGNMENT @ TOP if requested ==== #}
{% if assignment is defined and assignment %}
  {% if assignment_position in ['top', 'both'] %}
    {% set label = "Assignment" %}
    {% if ns.first %}
{{ label }}:
---

{{ assignment }}

      {% set ns.first = false %}
    {% else %}
---
{{ label }}:
---

{{ assignment }}

    {% endif %}
    {% set ns.assignment_printed = true %}
  {% endif %}
{% endif %}
{# ==== 3) INTENDED DESIGN (always “middle”, once) ==== #}
{% if intended_design is defined and intended_design %}
  {% if ns.first %}
Intended design:
---

{{ intended_design }}

    {% set ns.first = false %}
  {% else %}
---
Intended design:
---

{{ intended_design }}

  {% endif %}
{% endif %}
{# ==== 4) INTENDED USAGE ==== #}
{% if intended_usage is defined and intended_usage %}
  {% if ns.first %}
Intended usage:
---

{{ intended_usage }}

    {% set ns.first = false %}
  {% else %}
---
Intended usage:
---

{{ intended_usage }}

  {% endif %}
{% endif %}
{# ==== 5) 3RD PARTY DOCUMENTATION ==== #}
{% if third_party_documentation is defined and third_party_documentation %}
  {% if ns.first %}
3rd party documentation:
---

{{ third_party_documentation }}

    {% set ns.first = false %}
  {% else %}
---
3rd party documentation:
---

{{ third_party_documentation }}

  {% endif %}
{% endif %}
{# ==== 6) 1ST PARTY DOCUMENTATION ==== #}
{% if first_party_documentation is defined and first_party_documentation %}
  {% if ns.first %}
1st party documentation:
---

{{ first_party_documentation }}

    {% set ns.first = false %}
  {% else %}
---
1st party documentation:
---

{{ first_party_documentation }}

  {% endif %}
{% endif %}
{# ==== 7) 1ST PARTY CODE ==== #}
{% if first_party_code is defined and first_party_code %}
  {% if ns.first %}
1st party code:
---

{{ first_party_code }}

    {% set ns.first = false %}
  {% else %}
---
1st party code:
---

{{ first_party_code }}

  {% endif %}
{% endif %}
{# ==== 8) 1ST PARTY MISC FILES ==== #}
{% if first_party_misc_files is defined and first_party_misc_files %}
  {% if ns.first %}
1st party misc files:
---

{{ first_party_misc_files }}

    {% set ns.first = false %}
  {% else %}
---
1st party misc files:
---

{{ first_party_misc_files }}

  {% endif %}
{% endif %}
{# ==== 9) LOGS ==== #}
{% if logs is defined and logs %}
  {% if ns.first %}
Logs:
---

{{ logs }}

    {% set ns.first = false %}
  {% else %}
---
Logs:
---

{{ logs }}

  {% endif %}
{% endif %}
{# ==== 10) WORK ALREADY DONE ==== #}
{% if work_already_done is defined and work_already_done %}
  {% if ns.first %}
Work already done:
---

{{ work_already_done }}

    {% set ns.first = false %}
  {% else %}
---
Work already done:
---

{{ work_already_done }}

  {% endif %}
{% endif %}
{# ==== 11) WORK TO DO ==== #}
{% if work_to_do is defined and work_to_do %}
  {% if ns.first %}
Work to do:
---

{{ work_to_do }}

    {% set ns.first = false %}
  {% else %}
---
Work to do:
---

{{ work_to_do }}

  {% endif %}
{% endif %}
{# ==== 12) DIFF ==== #}
{% if diff is defined and diff %}
  {% if ns.first %}
Diff:
---

{{ diff }}

    {% set ns.first = false %}
  {% else %}
---
Diff:
---

{{ diff }}

  {% endif %}
{% endif %}
{#
  ==== 13) PRELUDE @ BOTTOM if requested ====
  If prelude_position == "bottom", this is the first time we’re printing it.
  If prelude_position == "both", we printed it in step 1, so now we repeat
  with label “Repeat of Prelude:”.
#}
{% if prelude is defined and prelude %}
  {% if prelude_position in ['bottom', 'both'] %}
    {% if ns.prelude_printed %}
      {% set label = "Repeat of Prelude" %}
    {% else %}
      {% set label = "Prelude" %}
    {% endif %}
    {% if ns.first %}
{{ label }}:
---

{{ prelude }}

      {% set ns.first = false %}
    {% else %}
---
{{ label }}:
---

{{ prelude }}

    {% endif %}
    {% set ns.prelude_printed = true %}
  {% endif %}
{% endif %}
{#
  ==== 14) ASSIGNMENT @ BOTTOM if requested ====
  Analogous logic to Prelude@bottom: if “both,” this is the second time,
  so label “Repeat of Assignment:”.
#}
{% if assignment is defined and assignment %}
  {% if assignment_position in ['bottom', 'both'] %}
    {% if ns.assignment_printed %}
      {% set label = "Repeat of Assignment" %}
    {% else %}
      {% set label = "Assignment" %}
    {% endif %}
    {% if ns.first %}
{{ label }}:
---

{{ assignment }}

      {% set ns.first = false %}
    {% else %}
---
{{ label }}:
---

{{ assignment }}

    {% endif %}
    {% set ns.assignment_printed = true %}
  {% endif %}
{% endif %}
"""


class StringLoader(BaseLoader):
    def __init__(self, templates: Dict[str, str]):
        self.templates = templates

    def get_source(self, environment: Environment, template: str) -> tuple[str, str | None, callable]:
        if template in self.templates:
            source = self.templates[template]
            return source, None, lambda: True  # source, filename, uptodate func
        raise TemplateNotFound(template)


class PromptBuilder(Component):
    display_name = "Prompt Builder"
    description = (
        "Builds a structured prompt string from multiple optional inputs, "
        "using a Jinja2 template for formatting. "
        "Supports DataFrame and list/dict inputs."
    )
    icon = "file-text"  # Or consider "edit", "align-left", "list-plus"
    name = "PromptBuilder"
    beta: bool = True

    inputs = [
        MultilineInput(name="prelude", display_name="Prelude", info="Opening text for the prompt.", value=""),
        MultilineInput(name="assignment", display_name="Assignment", info="Description of the task or assignment.", value=""),
        MultilineInput(name="intended_design", display_name="Intended Design", info="Details about the intended design.", value=""),
        MultilineInput(name="intended_usage", display_name="Intended Usage", info="How the subject is intended to be used.", value=""),
        MultilineInput(name="third_party_documentation", display_name="3rd Party Documentation", input_types=["Text", "DataFrame", "Data"], info="External documentation. DataFrame will be formatted with file paths.", value=""),
        MultilineInput(name="first_party_documentation", display_name="1st Party Documentation", input_types=["Text", "DataFrame", "Data"], info="Internal documentation. DataFrame will be formatted.", value=""),
        MultilineInput(name="first_party_code", display_name="1st Party Code", input_types=["Text", "DataFrame", "Data"], info="Relevant internal source code. DataFrame will be formatted.", value=""),
        MultilineInput(name="first_party_misc_files", display_name="1st Party Misc Files", input_types=["Text", "DataFrame", "Data"], info="Other relevant internal files. DataFrame will be formatted.", value=""),
        MultilineInput(name="logs", display_name="Logs", input_types=["Text", "DataFrame", "Data"], info="Log outputs. DataFrame will be formatted.", value=""),
        MultilineInput(name="work_already_done", display_name="Work Already Done", input_types=["Text", "DataFrame", "Data", "dict", "list"], info="Summary of work completed. Lists/Dicts will be bulleted.", value=""),
        MultilineInput(name="work_to_do", display_name="Work To Do", input_types=["Text", "DataFrame", "Data", "dict", "list"], info="Summary of pending work. Lists/Dicts will be bulleted.", value=""),
        MultilineInput(name="diff", display_name="Diff", info="A code diff or textual changes.", value=""),
        DropdownInput(name="prelude_position", display_name="Prelude Position", options=["top", "bottom", "both"], info="Position of the Prelude text.", value="top"),
        DropdownInput(name="assignment_position", display_name="Assignment Position", options=["top", "bottom", "both"], info="Position of the Assignment text.", value="bottom"),
    ]

    outputs = [
        Output(display_name="Prompt", name="report", method="build_report", info="The generated prompt string.")
    ]

    def _format_data_input(self, data_input_value: Any) -> str:
        if not data_input_value:
            return ""
        if isinstance(data_input_value, str):
            return data_input_value

        items_to_format: List[Data] = []
        if isinstance(data_input_value, DataFrame):
            items_to_format.extend(data_input_value.to_data_list())
        elif isinstance(data_input_value, Data):
            items_to_format.append(data_input_value)
        elif isinstance(data_input_value, list):  # Expected list of Data objects
            for item in data_input_value:
                if isinstance(item, Data):
                    items_to_format.append(item)
                elif isinstance(item, DataFrame): # Should be rare if data_types are restrictive
                    items_to_format.extend(item.to_data_list())
        else: # Not a string, DataFrame, Data, or list of Data
            return str(data_input_value) # Fallback to string representation

        if not items_to_format:
            return ""

        formatted_parts: List[str] = []
        for item_data in items_to_format:
            path = item_data.data.get("relative_path") or item_data.data.get("file_path")
            raw_content = str(item_data.get_text() or "")
            
            # Skip empty data unless it has a path (to indicate an empty file)
            if not raw_content.strip() and not path:
                continue

            if path:
                marker = "\n---\n\n" if path.lower().endswith(".md") else "```\n"
                # .lstrip("\n").rstrip("\n") from original format_directory_data
                cleaned_content = raw_content.lstrip("\n").rstrip("\n")
                block = f"`{path}`:\n{marker}{cleaned_content}"
                if cleaned_content:  # Add newline only if there's content
                    block += "\n"
                block += marker.rstrip() # Avoid double newlines from marker
                formatted_parts.append(block)
            else:
                formatted_parts.append(raw_content.strip())
        
        return "\n\n".join(formatted_parts)

    def _format_list_input(self, list_input_value: Any) -> str:
        if not list_input_value:
            return ""

        items_to_process: Union[List[Any], Dict[Any, Any]]
        
        if isinstance(list_input_value, str):
            if not list_input_value.strip(): # Handles string with only whitespace
                return ""
            try:
                # Attempt to parse string as JSON list or dict
                parsed_json = json.loads(list_input_value)
                if isinstance(parsed_json, (list, dict)):
                    items_to_process = parsed_json
                else: # Valid JSON but not list/dict, treat as plain string
                    return str(list_input_value)
            except json.JSONDecodeError: # Not a JSON string, treat as plain string
                return str(list_input_value)
        elif isinstance(list_input_value, (list, dict)):
            items_to_process = list_input_value
        elif isinstance(list_input_value, Data):
            data_content = list_input_value.get_data()
            if isinstance(data_content, (list, dict)):
                items_to_process = data_content
            else:
                text_content = list_input_value.get_text()
                if text_content:
                    try:
                        parsed_json = json.loads(text_content)
                        if isinstance(parsed_json, (list, dict)):
                            items_to_process = parsed_json
                        else:
                            items_to_process = [text_content]
                    except json.JSONDecodeError:
                        items_to_process = [text_content]
                else:
                    items_to_process = [] # Empty data
        elif isinstance(list_input_value, DataFrame):
            df_list = list_input_value.to_dict(orient='split')['data']
            header = list_input_value.to_dict(orient='split')['columns']
            items_to_process = ["DataFrame:", header] + df_list # Add a label and header
        else:
            items_to_process = [str(list_input_value)] # Fallback

        if not items_to_process:
            return ""

        def format_recursive(items: Union[List[Any], Dict[Any, Any]], indent_level: int) -> List[str]:
            lines: List[str] = []
            indent_space = "  " * indent_level
            
            if isinstance(items, dict):
                # Format dict as sub-bullets
                for key, value in items.items():
                    if isinstance(value, (list, dict)):
                        lines.append(f"{indent_space}- {key}:")
                        lines.extend(format_recursive(value, indent_level + 1))
                    else:
                        lines.append(f"{indent_space}- {key}: {str(value)}")
            elif isinstance(items, list):
                 for item in items:
                    if isinstance(item, (list, dict)):
                        # Add a generic marker for nested lists/dicts if not directly handled by dict iteration
                        # lines.append(f"{indent_space}- (Collection):") # Or rely on dict formatting above
                        lines.extend(format_recursive(item, indent_level + (0 if isinstance(item, dict) and indent_level == 0 else 1) )) # Avoid double indent for top-level dict
                    else:
                        lines.append(f"{indent_space}- {str(item)}")
            return lines

        # Initial call, handle if items_to_process is a dict directly
        if isinstance(items_to_process, dict):
            formatted_lines = format_recursive(items_to_process, 0)
        else: # It's a list (or was converted to a list of one item)
            formatted_lines = format_recursive(items_to_process, 0)
            
        return "\n".join(formatted_lines)

    def build_report(self) -> Message:
        context = {
            "prelude": str(self.prelude or ""),
            "assignment": str(self.assignment or ""),
            "intended_design": str(self.intended_design or ""),
            "intended_usage": str(self.intended_usage or ""),
            "third_party_documentation": self._format_data_input(self.third_party_documentation),
            "first_party_documentation": self._format_data_input(self.first_party_documentation),
            "first_party_code": self._format_data_input(self.first_party_code),
            "first_party_misc_files": self._format_data_input(self.first_party_misc_files),
            "logs": self._format_data_input(self.logs),
            "work_already_done": self._format_list_input(self.work_already_done),
            "work_to_do": self._format_list_input(self.work_to_do),
            "diff": str(self.diff or ""),
            "prelude_position": str(self.prelude_position or "top"),
            "assignment_position": str(self.assignment_position or "bottom"),
        }

        jinja_env = Environment(
            loader=StringLoader({"complete_report.jinja2": COMPLETE_REPORT_TEMPLATE}),
            autoescape=select_autoescape(enabled_extensions=('html', 'xml')), # Keep default, though not critical for text
            extensions=['jinja2.ext.do'], # For `{% do ns.var = ... %}`
            trim_blocks=True, # Helpful for template readability
            lstrip_blocks=True  # Helpful for template readability
        )
        template = jinja_env.get_template("complete_report.jinja2")
        
        try:
            rendered_report = template.render(context)
        except Exception as e:
            # Log error or set status, return error message
            self.status = f"Error rendering template: {str(e)}"
            return Message(text=f"Error generating report: {str(e)}")

        self.status = "Report generated successfully."
        return Message(text=rendered_report)