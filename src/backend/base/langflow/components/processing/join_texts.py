from langflow.custom import Component
from langflow.io import MessageTextInput, MultilineInput, Output
from langflow.schema.message import Message


class JoinTextsComponent(Component):
    display_name = "Join Texts"
    description = "Join multiple text inputs into a single text using a delimiter."
    icon = "merge"
    name = "JoinTexts"

    inputs = [
        MessageTextInput(
            name="texts",
            display_name="Texts",
            info="List of texts to concatenate.",
            is_list=True,
        ),
        MultilineInput(
            name="delimiter",
            display_name="Delimiter",
            info="Delimiter used to join texts.",
            value="\n",
        ),
    ]

    outputs = [
        Output(display_name="Combined Text", name="combined_text", method="join_texts"),
    ]

    def join_texts(self) -> Message:
        text_list = self.texts or []
        delimiter = self.delimiter
        processed_texts = [text.text if isinstance(text, Message) else str(text) for text in text_list]
        combined = delimiter.join(processed_texts)
        self.status = combined
        return Message(text=combined)
