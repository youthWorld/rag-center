from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.dataset import DatasetValidationError, load_and_validate_dataset  # noqa: E402
from app.evaluation.experiment import (  # noqa: E402
    ExperimentValidationError,
    load_and_validate_experiment,
)
from app.evaluation.metrics import score_run  # noqa: E402
from app.evaluation.report import write_report  # noqa: E402
from app.evaluation.runner import (  # noqa: E402
    EvaluationRunError,
    EvaluationRunner,
    HttpRetrievalClient,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run unified retrieval evaluations.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_dataset_parser = subparsers.add_parser(
        "validate-dataset", help="Validate a versioned Golden Set."
    )
    validate_dataset_parser.add_argument("--dataset", type=Path, required=True)

    validate_experiment_parser = subparsers.add_parser(
        "validate-experiment", help="Validate a baseline/candidate experiment."
    )
    validate_experiment_parser.add_argument("--experiment", type=Path, required=True)

    run_parser = subparsers.add_parser(
        "run", help="Run baseline and candidate retrieval requests."
    )
    run_parser.add_argument("--experiment", type=Path, required=True)
    run_parser.add_argument(
        "--base-url",
        default=os.getenv("EVAL_BASE_URL", DEFAULT_BASE_URL),
        help="RAG Center service URL.",
    )
    run_parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "eval" / "results",
    )
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--output-suffix")

    score_parser = subparsers.add_parser(
        "score", help="Score saved raw contexts without retrieving again."
    )
    score_parser.add_argument("--run-dir", type=Path, required=True)

    report_parser = subparsers.add_parser(
        "report", help="Render a report from saved metrics without external calls."
    )
    report_parser.add_argument("--run-dir", type=Path, required=True)
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
    if args.command == "validate-experiment":
        try:
            experiment = load_and_validate_experiment(args.experiment)
        except ExperimentValidationError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(
            f"experiment valid: id={experiment['experiment_id']} "
            f"baseline={experiment['baseline']['label']} "
            f"candidate={experiment['candidate']['label']} "
            f"changed_fields={','.join(experiment['changed_fields'])} "
            f"primary_metric={experiment['primary_metric']} "
            f"concurrency={experiment['concurrency']} "
            f"warmup_cases={experiment['warmup_cases']}"
        )
        return 0
    if args.command == "run":
        try:
            experiment = load_and_validate_experiment(args.experiment)
            dataset_path = _resolve_project_path(experiment["dataset"])
            dataset = load_and_validate_dataset(dataset_path)
            api_key = os.getenv("EVAL_API_KEY") or os.getenv("RAG_CENTER_API_KEY")
            if not api_key:
                raise EvaluationRunError(
                    "EVAL_API_KEY or RAG_CENTER_API_KEY must be set in the environment"
                )
            os.environ["RAGAS_DO_NOT_TRACK"] = "true"
            run_dir = asyncio.run(
                _run_retrievals(
                    experiment=experiment,
                    dataset=dataset,
                    api_key=api_key,
                    base_url=args.base_url,
                    output_root=args.output_root,
                    limit=args.limit,
                    output_suffix=args.output_suffix,
                )
            )
            comparison = score_run(run_dir)
            report_path = write_report(run_dir)
        except (
            DatasetValidationError,
            ExperimentValidationError,
            EvaluationRunError,
            OSError,
            ValueError,
        ) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(
            f"evaluation complete: run_dir={run_dir} "
            f"verdict={comparison['verdict']} report={report_path}"
        )
        return 0
    if args.command == "score":
        try:
            comparison = score_run(args.run_dir)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(
            f"scoring complete: run_dir={args.run_dir} verdict={comparison['verdict']}"
        )
        return 0
    if args.command == "report":
        try:
            report_path = write_report(args.run_dir)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"report complete: {report_path}")
        return 0

    raise AssertionError(f"unsupported command: {args.command}")


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


async def _run_retrievals(
    *,
    experiment: dict,
    dataset: dict,
    api_key: str,
    base_url: str,
    output_root: Path,
    limit: int | None,
    output_suffix: str | None,
) -> Path:
    async with HttpRetrievalClient(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=experiment["timeout_seconds"],
    ) as client:
        runner = EvaluationRunner(
            project_root=PROJECT_ROOT,
            experiment=experiment,
            dataset=dataset,
            client=client,
            output_root=output_root,
            limit=limit,
            output_suffix=output_suffix,
        )
        return await runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
