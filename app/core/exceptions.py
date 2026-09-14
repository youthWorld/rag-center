from typing import Any


class AppError(Exception):
    """An expected application error that can be returned to an API caller."""

    def __init__(
        self,
        message: str,
        *,
        code: int,
        status_code: int = 400,
        data: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.data = data


class KnowledgeBaseNotFoundError(AppError):
    def __init__(self) -> None:
        super().__init__(
            "knowledge base not found",
            code=40401,
            status_code=404,
        )


class DocumentIndexingError(AppError):
    def __init__(self, document_id: str, message: str) -> None:
        super().__init__(
            f"document indexing failed: {message}",
            code=50001,
            status_code=500,
            data={"document_id": document_id, "status": 2},
        )
