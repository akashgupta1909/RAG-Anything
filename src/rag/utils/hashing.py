"""Stable identifiers and content hashes."""

import hashlib
import uuid

# Fixed namespace so the same input always maps to the same point id.
_NAMESPACE = uuid.UUID("6f1c2a52-3b7e-4d0f-9a51-2a3c5e7d9b10")


def sha256(text: str | bytes) -> str:
    data = text.encode("utf-8") if isinstance(text, str) else text
    return hashlib.sha256(data).hexdigest()


def stable_uuid(key: str) -> str:
    """Deterministic UUID for ``key`` (Qdrant point ids must be UUIDs or ints)."""
    return str(uuid.uuid5(_NAMESPACE, key))
