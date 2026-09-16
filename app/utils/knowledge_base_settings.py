from typing import Any


def validate_knowledge_base_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Validate the settings shape while preserving the caller's full JSON object."""

    synonyms = settings.get("synonyms")
    if synonyms is None:
        return dict(settings)
    if not isinstance(synonyms, list):
        raise ValueError("settings.synonyms must be an array")

    for index, group in enumerate(synonyms):
        if not isinstance(group, dict):
            raise ValueError(f"settings.synonyms[{index}] must be an object")
        _validate_terms(group.get("terms"), f"settings.synonyms[{index}].terms")
        _validate_terms(group.get("expand"), f"settings.synonyms[{index}].expand")
    return dict(settings)


def _validate_terms(value: Any, field_name: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty array")
    if any(not isinstance(term, str) or not term.strip() for term in value):
        raise ValueError(f"{field_name} must contain non-empty strings")
