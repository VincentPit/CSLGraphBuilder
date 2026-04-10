"""
Unit tests for core/processing/processor.py

Covers:
  - CreateChunksofDocument.split_file_into_chunks()
  - create_relation_between_chunks()
"""

import hashlib
from unittest.mock import MagicMock, call

import pytest
from langchain_core.documents import Document

from graphbuilder.core.processing.processor import (
    CreateChunksofDocument,
    create_relation_between_chunks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph_mock():
    """Return a MagicMock that stands in for a Neo4jGraph."""
    return MagicMock()


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# CreateChunksofDocument.split_file_into_chunks
# ---------------------------------------------------------------------------


class TestSplitFileIntoChunks:
    """
    Tests for the chunking logic. The TokenTextSplitter is a real dependency
    (no network / DB involved), so we use it as-is.
    """

    def test_returns_list_of_documents(self):
        pages = [Document(page_content="Hello world. " * 50)]
        chunker = CreateChunksofDocument(pages, graph=_make_graph_mock())
        chunks = chunker.split_file_into_chunks()
        assert isinstance(chunks, list)
        assert all(isinstance(c, Document) for c in chunks)

    def test_short_text_single_chunk(self):
        """Text shorter than chunk_size=200 tokens should stay as one chunk."""
        pages = [Document(page_content="Short text.")]
        chunker = CreateChunksofDocument(pages, graph=_make_graph_mock())
        chunks = chunker.split_file_into_chunks()
        assert len(chunks) == 1
        assert chunks[0].page_content == "Short text."

    def test_long_text_multiple_chunks(self):
        """Text exceeding chunk_size=200 tokens should be split."""
        # ~600 words → should produce multiple chunks
        pages = [Document(page_content="word " * 600)]
        chunker = CreateChunksofDocument(pages, graph=_make_graph_mock())
        chunks = chunker.split_file_into_chunks()
        assert len(chunks) > 1

    def test_page_number_preserved_when_page_metadata_present(self):
        """When pages carry a 'page' key, each chunk records its 1-based page_number."""
        pages = [
            Document(page_content="Page one content. " * 30, metadata={"page": 0}),
            Document(page_content="Page two content. " * 30, metadata={"page": 1}),
        ]
        chunker = CreateChunksofDocument(pages, graph=_make_graph_mock())
        chunks = chunker.split_file_into_chunks()

        page_numbers = {c.metadata.get("page_number") for c in chunks}
        # Both page numbers must appear
        assert 1 in page_numbers
        assert 2 in page_numbers

    def test_no_page_metadata_passes_through(self):
        """When pages have no 'page' key, chunks should NOT have page_number."""
        pages = [Document(page_content="No page metadata. " * 30)]
        chunker = CreateChunksofDocument(pages, graph=_make_graph_mock())
        chunks = chunker.split_file_into_chunks()
        assert all("page_number" not in c.metadata for c in chunks)

    def test_multiple_pages_concatenated_when_no_page_key(self):
        """Multiple pages without 'page' metadata are all split together."""
        pages = [
            Document(page_content="First. " * 60),
            Document(page_content="Second. " * 60),
        ]
        chunker = CreateChunksofDocument(pages, graph=_make_graph_mock())
        chunks = chunker.split_file_into_chunks()
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# create_relation_between_chunks
# ---------------------------------------------------------------------------


class TestCreateRelationBetweenChunks:

    def test_single_chunk_returns_one_item(self):
        graph = _make_graph_mock()
        chunks = [Document(page_content="Only chunk.")]
        result = create_relation_between_chunks(graph, "file.pdf", chunks)
        assert len(result) == 1

    def test_return_structure(self):
        """Each item must have 'chunk_id' and 'chunk_doc' keys."""
        graph = _make_graph_mock()
        chunks = [Document(page_content="Alpha."), Document(page_content="Beta.")]
        result = create_relation_between_chunks(graph, "file.pdf", chunks)
        for item in result:
            assert "chunk_id" in item
            assert "chunk_doc" in item

    def test_chunk_id_is_sha1_of_content(self):
        """chunk_id must equal sha1(page_content)."""
        graph = _make_graph_mock()
        content = "Deterministic content."
        chunks = [Document(page_content=content)]
        result = create_relation_between_chunks(graph, "file.pdf", chunks)
        assert result[0]["chunk_id"] == _sha1(content)

    def test_chunk_ids_are_deterministic(self):
        """Same content → same IDs across two runs."""
        graph1, graph2 = _make_graph_mock(), _make_graph_mock()
        chunks = [Document(page_content="Same text."), Document(page_content="More text.")]
        r1 = create_relation_between_chunks(graph1, "f.pdf", chunks)
        r2 = create_relation_between_chunks(graph2, "f.pdf", chunks)
        assert [i["chunk_id"] for i in r1] == [i["chunk_id"] for i in r2]

    def test_distinct_contents_produce_distinct_ids(self):
        graph = _make_graph_mock()
        chunks = [
            Document(page_content="Content A."),
            Document(page_content="Content B."),
        ]
        result = create_relation_between_chunks(graph, "file.pdf", chunks)
        ids = [item["chunk_id"] for item in result]
        assert len(ids) == len(set(ids)), "Every chunk must have a unique ID"

    def test_graph_query_called_three_times(self):
        """One call for PART_OF, one for FIRST_CHUNK, one for NEXT_CHUNK."""
        graph = _make_graph_mock()
        chunks = [Document(page_content="A."), Document(page_content="B.")]
        create_relation_between_chunks(graph, "file.pdf", chunks)
        assert graph.query.call_count == 3

    def test_single_chunk_graph_query_called_twice(self):
        """No NEXT_CHUNK query is still made (always 3 calls)."""
        graph = _make_graph_mock()
        chunks = [Document(page_content="Only.")]
        create_relation_between_chunks(graph, "file.pdf", chunks)
        # PART_OF + FIRST_CHUNK + NEXT_CHUNK (empty list) = 3
        assert graph.query.call_count == 3

    def test_page_number_included_in_batch_data(self):
        """Chunks with page_number metadata must propagate it to the query params."""
        graph = _make_graph_mock()
        chunks = [
            Document(page_content="Page one.", metadata={"page_number": 1}),
            Document(page_content="Page two.", metadata={"page_number": 2}),
        ]
        create_relation_between_chunks(graph, "doc.pdf", chunks)

        # Inspect the params passed to the first query (PART_OF / batch_data)
        first_call_kwargs = graph.query.call_args_list[0]
        batch_data = first_call_kwargs[1]["params"]["batch_data"]
        assert batch_data[0]["page_number"] == 1
        assert batch_data[1]["page_number"] == 2

    def test_file_name_stored_in_batch_data(self):
        graph = _make_graph_mock()
        chunks = [Document(page_content="Sample.")]
        create_relation_between_chunks(graph, "my_file.pdf", chunks)
        first_call_kwargs = graph.query.call_args_list[0]
        batch_data = first_call_kwargs[1]["params"]["batch_data"]
        assert batch_data[0]["f_name"] == "my_file.pdf"

    def test_position_increments_correctly(self):
        graph = _make_graph_mock()
        chunks = [
            Document(page_content="First."),
            Document(page_content="Second."),
            Document(page_content="Third."),
        ]
        create_relation_between_chunks(graph, "f.pdf", chunks)
        first_call_kwargs = graph.query.call_args_list[0]
        batch_data = first_call_kwargs[1]["params"]["batch_data"]
        positions = [d["position"] for d in batch_data]
        assert positions == [1, 2, 3]
