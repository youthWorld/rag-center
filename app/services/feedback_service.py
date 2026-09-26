from uuid import NAMESPACE_URL, uuid5

from app.core.config import Settings
from app.core.exceptions import (
    FeedbackLogMismatchError,
    FeedbackScoreInvalidError,
    FeedbackUnavailableError,
)
from app.observability.langfuse_client import get_langfuse_client
from app.schemas.feedback import FeedbackData, FeedbackRequest


class FeedbackService:
    def __init__(
        self,
        *,
        settings: Settings,
    ) -> None:
        self.settings = settings

    async def submit(self, request: FeedbackRequest, *, tenant_id: str) -> FeedbackData:
        if request.score < 1 or request.score > 5:
            raise FeedbackScoreInvalidError()

        client = get_langfuse_client(self.settings)
        if client is None:
            raise FeedbackUnavailableError(
                internal_message="Langfuse is disabled or unavailable"
            )

        try:
            response = client.fetch_trace(request.trace_id)
            trace = getattr(response, "data", response)
            metadata = self._read_field(trace, "metadata")
            if (
                trace is None
                or not isinstance(metadata, dict)
                or metadata.get("tenant_id") != tenant_id
                or metadata.get("log_id") != request.log_id
            ):
                raise FeedbackLogMismatchError()
            feedback_id = self._resolve_feedback_id(trace, trace_id=request.trace_id)
        except FeedbackLogMismatchError:
            raise
        except Exception as exception:
            raise FeedbackUnavailableError(
                internal_message=(
                    "Langfuse feedback lookup failed: "
                    f"{str(exception) or type(exception).__name__}"
                )
            ) from exception

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

    @staticmethod
    def _resolve_feedback_id(trace: object, *, trace_id: str) -> str:
        scores = FeedbackService._read_field(trace, "scores") or ()
        for score in scores:
            name = FeedbackService._read_field(score, "name")
            if name != "user_feedback":
                continue
            score_id = FeedbackService._read_field(score, "id")
            if isinstance(score_id, str) and score_id.strip():
                return score_id.strip()
        return str(uuid5(NAMESPACE_URL, f"rag-center:{trace_id}:user_feedback"))

    @staticmethod
    def _read_field(value: object, field: str) -> object:
        if isinstance(value, dict):
            return value.get(field)
        return getattr(value, field, None)
