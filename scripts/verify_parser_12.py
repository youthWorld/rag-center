from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify JSON, DOCX, PDF, and source-file reparse ingestion."
    )
    parser.add_argument("--kb-id", required=True, help="Knowledge base used for the checks.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("RAG_CENTER_BASE_URL", "http://127.0.0.1:8000"),
        help="API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("RAG_CENTER_API_KEY", ""),
        help="Optional API key sent as a Bearer token.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "parser-12-samples",
        help="Directory for generated sample files.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    sample_dir = args.output_dir
    sample_dir.mkdir(parents=True, exist_ok=True)
    docx_path = sample_dir / "sample.docx"
    pdf_path = sample_dir / "sample.pdf"
    create_sample_docx(docx_path)
    create_sample_pdf(pdf_path)

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/"), headers=headers) as client:
        docx_id = await upload_file(client, args.kb_id, docx_path)
        await wait_for_success(client, docx_id, args.timeout)
        await assert_retrieval(client, args.kb_id, "DOCXPARSER12")
        print("PASS multipart DOCX upload, indexing, and retrieval")

        pdf_id = await upload_file(client, args.kb_id, pdf_path)
        await wait_for_success(client, pdf_id, args.timeout)
        await assert_retrieval(client, args.kb_id, "PDFPARSER12")
        print("PASS multipart PDF upload, indexing, and retrieval")

        reparse_response = await client.post(
            f"/api/v1/documents/{pdf_id}/reindex",
            params={"reparse": "true"},
        )
        ensure_success(reparse_response)
        await wait_for_success(client, pdf_id, args.timeout)
        await assert_retrieval(client, args.kb_id, "PDFPARSER12")
        print("PASS reparse=true rebuilds source-file content")

        markdown_response = await client.post(
            "/api/v1/documents/upload",
            json={
                "kb_id": args.kb_id,
                "title": "sample.md",
                "content": "# MARKDOWN_PARSER_12\n\nJSON upload regression content.",
            },
        )
        markdown_data = ensure_success(markdown_response)
        markdown_id = markdown_data["document_id"]
        await wait_for_success(client, markdown_id, args.timeout)
        await assert_retrieval(client, args.kb_id, "MARKDOWNPARSER12")
        print("PASS JSON Markdown upload regression")

    print("All parser 12 checks passed")
    return 0


async def upload_file(client: httpx.AsyncClient, kb_id: str, path: Path) -> str:
    with path.open("rb") as stream:
        response = await client.post(
            "/api/v1/documents/upload",
            data={"kb_id": kb_id},
            files={"file": (path.name, stream, _mime_type(path))},
        )
    data = ensure_success(response)
    return str(data["document_id"])


async def wait_for_success(
    client: httpx.AsyncClient,
    document_id: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        response = await client.get(f"/api/v1/documents/{document_id}")
        data = ensure_success(response)
        status = int(data["status"])
        if status == 1:
            return data
        if status == 2:
            raise RuntimeError(
                f"document {document_id} failed: {data.get('error_message') or 'unknown error'}"
            )
        await asyncio.sleep(1)
    raise TimeoutError(f"timed out waiting for document {document_id}")


async def assert_retrieval(
    client: httpx.AsyncClient,
    kb_id: str,
    keyword: str,
) -> None:
    response = await client.post(
        "/api/v1/rag/retrieve",
        json={"kb_id": kb_id, "user_id": "parser-12-verifier", "query": keyword},
    )
    data = ensure_success(response)
    rendered = str(data.get("retrieved_chunks", data))
    if keyword not in rendered:
        raise AssertionError(f"retrieval did not contain {keyword!r}: {rendered[:1000]}")


def create_sample_docx(path: Path) -> None:
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Parser 12 DOCX</w:t></w:r></w:p>
    <w:p><w:r><w:t>DOCXPARSER12 is a searchable Word document sample.</w:t></w:r></w:p>
    <w:tbl>
      <w:tr><w:tc><w:p><w:r><w:t>Key</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Value</w:t></w:r></w:p></w:tc></w:tr>
      <w:tr><w:tc><w:p><w:r><w:t>format</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>docx</w:t></w:r></w:p></w:tc></w:tr>
    </w:tbl>
    <w:sectPr><w:pgSz w:w="11906" w:h="16838"/></w:sectPr>
  </w:body>
</w:document>"""
    content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml"
    ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""
    relationships = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
    Target="word/document.xml"/>
</Relationships>"""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", relationships)
        archive.writestr("word/document.xml", document_xml)


def create_sample_pdf(path: Path) -> None:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PDF parsing is a core dependency; run uv sync before generating the sample"
        ) from exc

    document = fitz.open()
    try:
        page = document.new_page()
        page.insert_text((72, 72), "PDF_PARSER_12 is a searchable PDF document sample.")
        document.save(path)
    finally:
        document.close()


def ensure_success(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}") from exc
    if response.status_code >= 400 or payload.get("code") != 0:
        raise RuntimeError(f"HTTP {response.status_code}: {payload}")
    return payload.get("data") or {}


def _mime_type(path: Path) -> str:
    return {
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pdf": "application/pdf",
    }.get(path.suffix.lower(), "application/octet-stream")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        return asyncio.run(main_async(args))
    except Exception as exc:
        print(f"parser 12 verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
