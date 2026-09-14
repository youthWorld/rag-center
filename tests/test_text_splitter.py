import pytest

from app.utils.text_splitter import TextSplitter


def test_short_text_is_returned_as_one_chunk() -> None:
    splitter = TextSplitter(chunk_size=20, chunk_overlap=5)

    assert splitter.split_text("hello world") == ["hello world"]


def test_long_text_is_split_with_overlap() -> None:
    splitter = TextSplitter(chunk_size=10, chunk_overlap=3)

    chunks = splitter.split_text("0123456789 abcdefghij")

    assert len(chunks) >= 2
    assert all(0 < len(chunk) <= 10 for chunk in chunks)
    assert chunks[0][-3:] in chunks[1]


def test_blank_text_is_ignored() -> None:
    assert TextSplitter().split_text(" \n\t ") == []


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (10, 10), (10, 11), (10, -1)],
)
def test_invalid_splitter_configuration_is_rejected(chunk_size: int, chunk_overlap: int) -> None:
    with pytest.raises(ValueError):
        TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
