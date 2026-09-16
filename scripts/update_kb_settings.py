from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import session_factory  # noqa: E402
from app.repositories.knowledge_base_repository import KnowledgeBaseRepository  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace query-processing settings for a RAG Center knowledge base."
    )
    parser.add_argument(
        "--kb_id",
        dest="kb_id",
        required=True,
        metavar="KB_ID",
        help="Knowledge base ID.",
    )
    parser.add_argument("--settings-file", required=True, type=Path)
    return parser.parse_args()


def load_settings(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"settings file does not exist: {path}") from exc
    except OSError as exc:
        raise SystemExit(f"failed to read settings file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"settings file contains invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise SystemExit("settings file must contain a JSON object")
    return payload


async def update_kb_settings(kb_id: str, settings: dict[str, Any]) -> None:
    async with session_factory() as session:
        repository = KnowledgeBaseRepository(session)
        # The whole JSON object is replaced so the file remains the source of truth.
        knowledge_base = await repository.update_settings(kb_id=kb_id, settings=settings)
        if knowledge_base is None:
            raise SystemExit(f"knowledge base does not exist: {kb_id}")
        await session.commit()

    synonyms = settings.get("synonyms")
    synonym_group_count = len(synonyms) if isinstance(synonyms, list) else 0
    print(f"updated kb_id: {kb_id}")
    print(f"synonym groups: {synonym_group_count}")


def main() -> None:
    args = parse_args()
    settings = load_settings(args.settings_file)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(update_kb_settings(args.kb_id, settings))


if __name__ == "__main__":
    main()
