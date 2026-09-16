from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import session_factory  # noqa: E402
from app.repositories.tenant_repository import TenantRepository  # noqa: E402
from app.tenant.plan_presets import PLAN_PRESETS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update a RAG Center tenant plan.")
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--plan", required=True)
    return parser.parse_args()


async def update_tenant_plan(tenant_id: str, plan: str) -> None:
    normalized_plan = plan.strip().lower()
    if normalized_plan not in PLAN_PRESETS:
        valid_plans = ", ".join(PLAN_PRESETS)
        raise SystemExit(f"invalid plan: {plan}; expected one of: {valid_plans}")

    async with session_factory() as session:
        tenant = await TenantRepository(session).get_by_id(tenant_id)
        if tenant is None:
            raise SystemExit(
                f"tenant not found: {tenant_id}; run create_tenant first"
            )
        tenant.plan = normalized_plan
        await session.commit()
        await session.refresh(tenant)

    print(f"tenant plan updated: {tenant.id} -> {tenant.plan}")


def main() -> None:
    args = parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(update_tenant_plan(args.tenant_id, args.plan))


if __name__ == "__main__":
    main()
