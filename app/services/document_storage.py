from __future__ import annotations

import asyncio
from pathlib import Path

from app.core.logging import get_logger


class DocumentStorage:
    """Persist uploaded source files below one controlled storage root."""

    def __init__(self, root_path: str) -> None:
        self.root = Path(root_path).expanduser().resolve()
        self.logger = get_logger(__name__)

    @staticmethod
    def validate_component(value: str, *, label: str) -> str:
        if (
            not value
            or value in {".", ".."}
            or "\x00" in value
            or "/" in value
            or "\\" in value
            or ":" in value
        ):
            raise ValueError(f"{label} must be a single safe path component")
        return value

    @classmethod
    def normalize_filename(cls, filename: str) -> str:
        normalized = Path(filename.replace("\\", "/")).name
        if not normalized or normalized in {".", ".."}:
            raise ValueError("file name must not be blank")
        if len(normalized) > 255:
            raise ValueError("file name is too long")
        return cls.validate_component(normalized, label="file name")

    def path_for(self, *, tenant_id: str, document_id: str, filename: str) -> Path:
        tenant_component = self.validate_component(tenant_id, label="tenant_id")
        document_component = self.validate_component(document_id, label="document_id")
        filename_component = self.normalize_filename(filename)
        path = self.root / tenant_component / document_component / filename_component
        self._ensure_within_root(path)
        return path

    async def write_bytes(self, path: Path, file_bytes: bytes) -> None:
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_bytes, file_bytes)

    async def remove(
        self,
        *,
        tenant_id: str,
        document_id: str,
        source_file_path: str | None,
    ) -> None:
        if not source_file_path:
            return

        try:
            tenant_component = self.validate_component(tenant_id, label="tenant_id")
            document_component = self.validate_component(document_id, label="document_id")
            expected_directory = self.root / tenant_component / document_component
            candidate = Path(source_file_path).expanduser().resolve()
            expected_directory = expected_directory.resolve()
            candidate.relative_to(expected_directory)
            if candidate.parent != expected_directory:
                raise ValueError("source file is outside its document directory")
            self._ensure_within_root(candidate)
        except (OSError, ValueError) as exc:
            self.logger.warning(
                "BUSINESS_WARNING | event=document_source_cleanup_skipped | "
                "tenant_id=%s | document_id=%s | path=%s | reason=%s",
                tenant_id,
                document_id,
                source_file_path,
                exc,
            )
            return

        try:
            await asyncio.to_thread(candidate.unlink, missing_ok=True)
            await asyncio.to_thread(expected_directory.rmdir)
        except OSError as exc:
            self.logger.warning(
                "BUSINESS_WARNING | event=document_source_cleanup_failed | "
                "tenant_id=%s | document_id=%s | path=%s | reason=%s",
                tenant_id,
                document_id,
                candidate,
                exc,
            )

    def _ensure_within_root(self, path: Path) -> None:
        path.resolve().relative_to(self.root)
