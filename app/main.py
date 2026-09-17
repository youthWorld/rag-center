from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1.router import router as v1_router
from app.core.config import settings
from app.core.error_codes import ErrorCode
from app.core.exceptions import AppError
from app.core.logging import (
    configure_logging,
    get_logger,
    log_exception,
    request_logging_middleware,
)

logger = get_logger(__name__)


def create_app() -> FastAPI:
    configure_logging()
    application = FastAPI(
        title="RAG Center",
        version="0.1.0",
        description="A small, extensible RAG platform backend.",
    )
    application.state.settings = settings
    application.middleware("http")(request_logging_middleware)
    application.include_router(v1_router)

    @application.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.warning(
            "APP_ERROR | code=%s | status_code=%s | public_message=%s | "
            "internal_message=%s | context=%s",
            exc.code,
            exc.status_code,
            exc.message,
            exc.internal_message,
            exc.context,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_response(),
            headers={"X-Request-ID": getattr(request.state, "request_id", "-")},
        )

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = _serialize_validation_errors(exc.errors())
        if request.url.path == "/api/v1/rag/feedback" and _has_feedback_score_range_error(
            exc.errors()
        ):
            return JSONResponse(
                status_code=400,
                content={
                    "code": ErrorCode.FEEDBACK_SCORE_INVALID.code,
                    "msg": ErrorCode.FEEDBACK_SCORE_INVALID.message,
                    "data": None,
                },
                headers={"X-Request-ID": getattr(request.state, "request_id", "-")},
            )
        logger.warning(
            "REQUEST_VALIDATION_ERROR | url=%s | errors=%s",
            request.url,
            errors,
        )
        return JSONResponse(
            status_code=400,
            content={
                "code": ErrorCode.REQUEST_VALIDATION_ERROR.code,
                "msg": ErrorCode.REQUEST_VALIDATION_ERROR.message,
                "data": {"errors": errors},
            },
            headers={"X-Request-ID": getattr(request.state, "request_id", "-")},
        )

    @application.exception_handler(StarletteHTTPException)
    async def handle_http_error(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        error_code = {
            401: ErrorCode.UNAUTHORIZED,
            403: ErrorCode.FORBIDDEN,
            404: ErrorCode.NOT_FOUND,
            405: ErrorCode.METHOD_ERROR,
        }.get(exc.status_code, ErrorCode.API_REQUEST_ERROR)
        message = (
            error_code.message
            if exc.status_code in {401, 403, 404, 405} or exc.status_code >= 500
            else exc.detail
            if isinstance(exc.detail, str)
            else error_code.message
        )
        logger.warning(
            "HTTP_ERROR | status_code=%s | code=%s | detail=%s | url=%s",
            exc.status_code,
            error_code.code,
            exc.detail,
            request.url,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": error_code.code, "msg": message, "data": None},
            headers={"X-Request-ID": getattr(request.state, "request_id", "-")},
        )

    @application.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        log_exception(
            exc,
            logger=logger,
            context={
                "method": request.method,
                "url": str(request.url),
                "request_id": getattr(request.state, "request_id", None),
            },
        )
        return JSONResponse(
            status_code=500,
            content={
                "code": ErrorCode.SERVER_ERROR.code,
                "msg": ErrorCode.SERVER_ERROR.message,
                "data": None,
            },
            headers={"X-Request-ID": getattr(request.state, "request_id", "-")},
        )

    return application


def _serialize_validation_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "loc": list(error.get("loc", [])),
            "msg": error.get("msg", "invalid value"),
            "type": error.get("type", "value_error"),
        }
        for error in errors
    ]


def _has_feedback_score_range_error(errors: list[dict[str, Any]]) -> bool:
    range_error_types = {
        "greater_than",
        "greater_than_equal",
        "less_than",
        "less_than_equal",
    }
    return any(
        error.get("type") in range_error_types
        and list(error.get("loc", []))[-1:] == ["score"]
        for error in errors
    )


app = create_app()
