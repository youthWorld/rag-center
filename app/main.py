from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.v1.router import router as v1_router
from app.core.exceptions import AppError
from app.core.logging import configure_logging


def create_app() -> FastAPI:
    configure_logging()
    application = FastAPI(
        title="RAG Center",
        version="0.1.0",
        description="A small, extensible RAG platform backend.",
    )
    application.include_router(v1_router)

    @application.exception_handler(AppError)
    async def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "msg": exc.message, "data": exc.data},
        )

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "code": 40000,
                "msg": "request validation failed",
                "data": {"errors": _serialize_validation_errors(exc.errors())},
            },
        )

    @application.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, _exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"code": 50000, "msg": "internal server error", "data": None},
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


app = create_app()
