from app.core.config import Settings
from app.core.exceptions import (
    FeedbackAlreadySubmittedError,
    FeedbackLogMismatchError,
    FeedbackScoreInvalidError,
    FeedbackUnavailableError,
)
from app.observability.langfuse_client import get_langfuse_client, has_trace_score
from app.repositories.retrieval_log_repository import RetrievalLogRepository
from app.schemas.feedback import FeedbackData, FeedbackRequest
from app.utils.id_generator import generate_id


class FeedbackService:
    def __init__(
        self,
        *,
        settings: Settings,
        retrieval_log_repository: RetrievalLogRepository,
    ) -> None:
        self.settings = settings
        self.retrieval_log_repository = retrieval_log_repository

    async def submit(self, request: FeedbackRequest, *, tenant_id: str) -> FeedbackData:
        if request.score < 1 or request.score > 5:
            raise FeedbackScoreInvalidError()

        if request.log_id is not None:
            retrieval_log = await self.retrieval_log_repository.get_by_id(
                log_id=request.log_id
            )
            if (
                retrieval_log is None
                or retrieval_log.tenant_id != tenant_id
                or retrieval_log.trace_id != request.trace_id
            ):
                raise FeedbackLogMismatchError()

        client = get_langfuse_client(self.settings)
        if client is None:
            raise FeedbackUnavailableError(
                internal_message="Langfuse is disabled or unavailable"
            )

        acquire_feedback_lock = getattr(
            self.retrieval_log_repository,
            "acquire_feedback_lock",
            None,
        )
        if acquire_feedback_lock is not None:
            await acquire_feedback_lock(trace_id=request.trace_id)

        try:
            if has_trace_score(
                client,
                trace_id=request.trace_id,
                name="user_feedback",
            ):
                raise FeedbackAlreadySubmittedError()
        except FeedbackAlreadySubmittedError:
            raise
        except Exception as exception:
            raise FeedbackUnavailableError(
                internal_message=(
                    "Langfuse feedback lookup failed: "
                    f"{str(exception) or type(exception).__name__}"
                )
            ) from exception

        feedback_id = generate_id()
        try:
            client.score(
                id=feedback_id,
                name="user_feedback",
                value=request.score,
                trace_id=request.trace_id,
                comment=request.comment,
            )
            client.flush()
        except Exception as exception:
            raise FeedbackUnavailableError(
                internal_message=(
                    "Langfuse feedback write failed: "
                    f"{str(exception) or type(exception).__name__}"
                )
            ) from exception

        return FeedbackData(
            feedback_id=feedback_id,
            trace_id=request.trace_id,
            log_id=request.log_id,
            score=request.score,
        )
