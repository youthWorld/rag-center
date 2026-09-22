from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.dataset import DatasetValidationError, load_and_validate_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run unified retrieval evaluations.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_dataset_parser = subparsers.add_parser(
        "validate-dataset", help="Validate a versioned Golden Set."
    )
    validate_dataset_parser.add_argument("--dataset", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "validate-dataset":
        try:
            dataset = load_and_validate_dataset(args.dataset)
        except DatasetValidationError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        basic_count = len(dataset["suites"]["basic"]["case_ids"])
        print(
            f"dataset valid: name={dataset['name']} version={dataset['version']} "
            f"basic_cases={basic_count}"
        )
        return 0

    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
