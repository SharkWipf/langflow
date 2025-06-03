from langflow.components.utilities import PromptBuilder

# Define a dictionary of all components, categorized
# This is a common pattern for component registration.
# Langflow might have a specific way to load these,
# but this structure helps in organizing and discovering components.

COMPONENTS_BY_CATEGORY = {
    "Utilities": [
        PromptBuilder,
    ],
    # Add other categories and their components here
    # e.g., "Text Processing": [SomeTextComponent],
}

# Flatten the list of all components for easier access if needed elsewhere
ALL_COMPONENTS = [
    component for category_components in COMPONENTS_BY_CATEGORY.values() for component in category_components
]

__all__ = ["COMPONENTS_BY_CATEGORY", "ALL_COMPONENTS"]