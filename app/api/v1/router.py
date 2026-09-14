from fastapi import APIRouter

from app.api.v1.routes import documents, knowledge_bases, rag

router = APIRouter(prefix="/api/v1")
router.include_router(knowledge_bases.router)
router.include_router(documents.router)
router.include_router(rag.router)
