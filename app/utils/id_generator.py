from uuid import uuid4


def generate_id() -> str:
    """Return the canonical string representation of an internal resource UUID."""
    return str(uuid4())
