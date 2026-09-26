from app.schemas.rag import RetrievedChunk
from app.services.research_candidate_fusion_service import ResearchCandidateFusionService
from app.services.research_limits import MAX_RESEARCH_CANDIDATES
from app.services.retrieve_once_service import RetrieveOnceResult


def _result(query_id: str, aspect_id: str, chunks: list[RetrievedChunk], round_number=1):
    return RetrieveOnceResult(
        query_id=query_id, query=f"query {query_id}", aspect_ids=[aspect_id],
        round=round_number, candidate_snapshot=[], retrieved_chunks=chunks,
        metadata={}, latency_ms=1,
    )


def _chunk(chunk_id: str, *, kb_id="kb-a", content="same body"):
    return RetrievedChunk(
        chunk_id=chunk_id, kb_id=kb_id, index_version="v2", document_id="doc",
        title="title", content=content, score=1.0,
    )


def test_fusion_merges_provenance_but_keeps_kb_scope() -> None:
    candidates = ResearchCandidateFusionService().merge([
        _result("Q1", "A1", [_chunk("shared"), _chunk("other", content="other")]),
        _result("Q2", "A2", [_chunk("shared"), _chunk("shared", kb_id="kb-b")], 2),
    ])
    same = next(item for item in candidates if item.kb_id == "kb-a" and item.chunk_id == "shared")
    assert same.aspect_ids == ["A1", "A2"]
    assert same.query_ids == ["Q1", "Q2"]
    assert same.rounds == [1, 2]
    assert abs(same.fusion_score - 2 / 61) < 1e-9
    assert any(item.kb_id == "kb-b" and item.chunk_id == "shared" for item in candidates)
    assert [item.candidate_id for item in candidates] == [
        f"C{i}" for i in range(1, len(candidates) + 1)
    ]


def test_fusion_reserves_a_slot_for_each_aspect_before_global_ranking() -> None:
    many = [_chunk(f"a{i}", content=f"body {i}") for i in range(40)]
    candidates = ResearchCandidateFusionService().merge([
        _result("Q1", "A1", many),
        _result("Q2", "A2", [_chunk("b1", content="only B")]),
    ])
    assert len(candidates) == MAX_RESEARCH_CANDIDATES
    assert candidates[0].chunk_id == "a0"
    assert candidates[1].chunk_id == "b1"
    assert any("A2" in item.aspect_ids for item in candidates)


def test_fusion_deduplicates_same_text_only_inside_the_same_kb_and_version() -> None:
    candidates = ResearchCandidateFusionService().merge([
        _result("Q1", "A1", [_chunk("first", content="Repeated  text")]),
        _result("Q2", "A2", [
            _chunk("second", content=" repeated text "),
            _chunk("third", kb_id="kb-b", content="Repeated text"),
        ]),
    ])
    assert len(candidates) == 2
    first = next(item for item in candidates if item.kb_id == "kb-a")
    assert first.chunk_id == "first"
    assert first.query_ids == ["Q1", "Q2"]
    assert first.aspect_ids == ["A1", "A2"]
    assert first.metadata["duplicate_chunk_ids"] == ["second"]
    assert next(item for item in candidates if item.kb_id == "kb-b").chunk_id == "third"
