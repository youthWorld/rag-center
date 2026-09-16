from app.providers.query.base import QueryContext, QueryProcessor, QueryProcessResult


class NoopQueryProcessor(QueryProcessor):
    async def process(
        self,
        result: QueryProcessResult,
        *,
        context: QueryContext,
    ) -> QueryProcessResult:
        del context
        return result
