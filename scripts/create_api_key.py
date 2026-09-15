from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.exc import IntegrityError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.auth import generate_api_key  # noqa: E402
from app.db.session import session_factory  # noqa: E402
from app.repositories.api_key_repository import ApiKeyRepository  # noqa: E402
from app.repositories.tenant_repository import TenantRepository  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a RAG Center API key.")
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--expires-at",
        help="Optional ISO-8601 expiration, for example 2026-12-31T23:59:59Z.",
    )
    return parser.parse_args()


def parse_expiration(value: str | None) -> datetime | None:
    if value is None:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    expires_at = datetime.fromisoformat(normalized)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    return expires_at


async def create_api_key(
    tenant_id: str,
    name: str,
    expires_at: datetime | None,
) -> None:
    async with session_factory() as session:
        tenant = await TenantRepository(session).get_by_id(tenant_id)
        if tenant is None:
            raise SystemExit(f"tenant does not exist: {tenant_id}")

        plaintext, key_hash, key_prefix = generate_api_key()
        repository = ApiKeyRepository(session)
        try:
            record = await repository.create(
                tenant_id=tenant_id,
                key_hash=key_hash,
                key_prefix=key_prefix,
                name=name,
                expires_at=expires_at,
            )
            await session.commit()
            await session.refresh(record)
        except IntegrityError as exc:
            await session.rollback()
            raise SystemExit("failed to create API key") from exc

    print(f"api key (shown once): {plaintext}")
    print(f"key prefix: {record.key_prefix}")


def main() -> None:
    args = parse_args()
    try:
        expires_at = parse_expiration(args.expires_at)
    except ValueError as exc:
        raise SystemExit(f"invalid --expires-at: {exc}") from exc
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(create_api_key(args.tenant_id, args.name, expires_at))


if __name__ == "__main__":
    main()
