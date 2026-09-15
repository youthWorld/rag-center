from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy.exc import IntegrityError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import session_factory  # noqa: E402
from app.repositories.tenant_repository import TenantRepository  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a RAG Center tenant.")
    parser.add_argument("--id", required=True, dest="tenant_id")
    parser.add_argument("--name", required=True)
    return parser.parse_args()


async def create_tenant(tenant_id: str, name: str) -> None:
    async with session_factory() as session:
        repository = TenantRepository(session)
        if await repository.get_by_id(tenant_id):
            raise SystemExit(f"tenant already exists: {tenant_id}")

        try:
            tenant = await repository.create(tenant_id=tenant_id, name=name)
            await session.commit()
            await session.refresh(tenant)
        except IntegrityError as exc:
            await session.rollback()
            raise SystemExit(f"tenant already exists: {tenant_id}") from exc

    print(f"tenant created: {tenant.id} ({tenant.name})")


def main() -> None:
    args = parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(create_tenant(args.tenant_id, args.name))


if __name__ == "__main__":
    main()
