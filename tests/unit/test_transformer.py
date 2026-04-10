"""
Unit tests for core/graph/transformer.py

Covers:
  - format_property_key()
  - map_to_base_node()
  - map_to_base_relationship()
  - _format_nodes()
  - _format_relationships()
  - _convert_to_graph_document()
  - LLMGraphTransformer.process_response() (via mocked chain)
"""

from unittest.mock import MagicMock, patch
from typing import Optional

import pytest
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document

from graphbuilder.core.graph.transformer import (
    LLMGraphTransformer,
    _convert_to_graph_document,
    _format_nodes,
    _format_relationships,
    format_property_key,
    map_to_base_node,
    map_to_base_relationship,
)


# ---------------------------------------------------------------------------
# format_property_key
# ---------------------------------------------------------------------------


class TestFormatPropertyKey:
    def test_single_word_lowercased(self):
        assert format_property_key("Name") == "name"

    def test_two_words_camel_case(self):
        assert format_property_key("first name") == "firstName"

    def test_three_words_camel_case(self):
        assert format_property_key("date of birth") == "dateOfBirth"

    def test_already_camel_case_single_word(self):
        assert format_property_key("id") == "id"

    def test_empty_string_returns_empty(self):
        assert format_property_key("") == ""


# ---------------------------------------------------------------------------
# map_to_base_node
# ---------------------------------------------------------------------------


class TestMapToBaseNode:
    def _make_simple_node(self, id: str, type: str, properties=None):
        node = MagicMock()
        node.id = id
        node.type = type
        node.properties = properties  # None or list of MagicMock with .key/.value
        return node

    def test_basic_node_no_properties(self):
        node = self._make_simple_node("Alice", "Person")
        result = map_to_base_node(node)
        assert isinstance(result, Node)
        assert result.id == "Alice"
        assert result.type == "Person"
        assert result.properties == {}

    def test_node_with_properties(self):
        prop = MagicMock()
        prop.key = "age"
        prop.value = "30"
        node = self._make_simple_node("Alice", "Person", properties=[prop])
        result = map_to_base_node(node)
        assert result.properties == {"age": "30"}

    def test_node_properties_key_formatted(self):
        prop = MagicMock()
        prop.key = "birth date"
        prop.value = "1990-01-01"
        node = self._make_simple_node("Bob", "Person", properties=[prop])
        result = map_to_base_node(node)
        assert "birthDate" in result.properties


# ---------------------------------------------------------------------------
# map_to_base_relationship
# ---------------------------------------------------------------------------


class TestMapToBaseRelationship:
    def _make_rel(self, src_id, src_type, tgt_id, tgt_type, rel_type):
        rel = MagicMock()
        rel.source_node_id = src_id
        rel.source_node_type = src_type
        rel.target_node_id = tgt_id
        rel.target_node_type = tgt_type
        rel.type = rel_type
        return rel

    def test_basic_relationship(self):
        rel = self._make_rel("Alice", "Person", "Microsoft", "Company", "WORKS_FOR")
        result = map_to_base_relationship(rel)
        assert isinstance(result, Relationship)
        assert result.source.id == "Alice"
        assert result.target.id == "Microsoft"
        assert result.type == "WORKS_FOR"

    def test_source_and_target_are_nodes(self):
        rel = self._make_rel("A", "T1", "B", "T2", "REL")
        result = map_to_base_relationship(rel)
        assert isinstance(result.source, Node)
        assert isinstance(result.target, Node)


# ---------------------------------------------------------------------------
# _format_nodes
# ---------------------------------------------------------------------------


class TestFormatNodes:
    def test_id_is_title_cased(self):
        nodes = [Node(id="john doe", type="person")]
        result = _format_nodes(nodes)
        assert result[0].id == "John Doe"

    def test_type_is_capitalized(self):
        nodes = [Node(id="Alice", type="person")]
        result = _format_nodes(nodes)
        assert result[0].type == "Person"

    def test_properties_preserved(self):
        nodes = [Node(id="Alice", type="Person", properties={"age": "30"})]
        result = _format_nodes(nodes)
        assert result[0].properties == {"age": "30"}

    def test_multiple_nodes(self):
        nodes = [Node(id="a", type="t1"), Node(id="b", type="t2")]
        result = _format_nodes(nodes)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _format_relationships
# ---------------------------------------------------------------------------


class TestFormatRelationships:
    def _make_rel(self, type_str: str) -> Relationship:
        src = Node(id="alice", type="person")
        tgt = Node(id="microsoft", type="company")
        return Relationship(source=src, target=tgt, type=type_str)

    def test_type_uppercased(self):
        rel = self._make_rel("works for")
        result = _format_relationships([rel])
        assert result[0].type == "WORKS_FOR"

    def test_spaces_replaced_with_underscores(self):
        rel = self._make_rel("has award")
        result = _format_relationships([rel])
        assert " " not in result[0].type

    def test_already_valid_type_unchanged(self):
        rel = self._make_rel("WORKS_FOR")
        result = _format_relationships([rel])
        assert result[0].type == "WORKS_FOR"

    def test_source_and_target_formatted(self):
        rel = self._make_rel("REL")
        result = _format_relationships([rel])
        assert result[0].source.id == "Alice"   # title-cased
        assert result[0].target.id == "Microsoft"


# ---------------------------------------------------------------------------
# _convert_to_graph_document
# ---------------------------------------------------------------------------


class TestConvertToGraphDocument:
    def test_valid_parsed_schema(self):
        node = MagicMock()
        node.id = "Alice"
        node.type = "Person"
        node.properties = None

        rel = MagicMock()
        rel.source_node_id = "Alice"
        rel.source_node_type = "Person"
        rel.target_node_id = "Microsoft"
        rel.target_node_type = "Company"
        rel.type = "WORKS_FOR"

        parsed = MagicMock()
        parsed.nodes = [node]
        parsed.relationships = [rel]

        raw_schema = {"parsed": parsed, "raw": None}
        nodes, relationships = _convert_to_graph_document(raw_schema)

        assert len(nodes) == 1
        assert nodes[0].id == "Alice"
        assert len(relationships) == 1
        assert relationships[0].type == "WORKS_FOR"

    def test_empty_parsed_schema(self):
        parsed = MagicMock()
        parsed.nodes = []
        parsed.relationships = []
        raw_schema = {"parsed": parsed, "raw": None}
        nodes, rels = _convert_to_graph_document(raw_schema)
        assert nodes == []
        assert rels == []

    def test_none_parsed_falls_back_to_raw_json(self):
        """When 'parsed' is None/falsy but raw has valid tool_calls JSON."""
        argument_json = {
            "nodes": [{"id": "Alice", "type": "Person"}],
            "relationships": [
                {
                    "source_node_id": "Alice",
                    "source_node_type": "Person",
                    "target_node_id": "Bob",
                    "target_node_type": "Person",
                    "type": "KNOWS",
                }
            ],
        }
        import json

        raw_msg = MagicMock()
        raw_msg.additional_kwargs = {
            "tool_calls": [{"function": {"arguments": json.dumps(argument_json)}}]
        }
        raw_schema = {"parsed": None, "raw": raw_msg}
        nodes, rels = _convert_to_graph_document(raw_schema)
        assert len(nodes) == 1
        assert len(rels) == 1

    def test_unparseable_returns_empty(self):
        """When both parsed and raw JSON fail, return empty lists."""
        raw_msg = MagicMock()
        raw_msg.additional_kwargs = {}  # no tool_calls key
        raw_schema = {"parsed": None, "raw": raw_msg}
        nodes, rels = _convert_to_graph_document(raw_schema)
        assert nodes == []
        assert rels == []


# ---------------------------------------------------------------------------
# LLMGraphTransformer
# ---------------------------------------------------------------------------


class TestLLMGraphTransformer:
    def _make_mock_llm(self):
        """Build a mock LLM that supports with_structured_output."""
        llm = MagicMock()
        structured = MagicMock()
        llm.with_structured_output.return_value = structured
        return llm

    def test_instantiation_with_function_call(self):
        llm = self._make_mock_llm()
        transformer = LLMGraphTransformer(llm=llm)
        assert transformer._function_call is True

    def test_instantiation_falls_back_when_structured_output_unsupported(self):
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError
        # json_repair not installed in test env — patch it so we don't fail on import
        with patch.dict("sys.modules", {"json_repair": MagicMock()}):
            transformer = LLMGraphTransformer(llm=llm)
        assert transformer._function_call is False

    def test_process_response_returns_graph_document(self):
        """process_response should return a GraphDocument for the given Document."""
        llm = self._make_mock_llm()
        transformer = LLMGraphTransformer(llm=llm)

        # The chain is prompt | structured_llm — mock its invoke
        node_mock = MagicMock()
        node_mock.id = "Alice"
        node_mock.type = "Person"
        node_mock.properties = None

        parsed_mock = MagicMock()
        parsed_mock.nodes = [node_mock]
        parsed_mock.relationships = []

        transformer.chain = MagicMock()
        transformer.chain.invoke.return_value = {"parsed": parsed_mock, "raw": None}

        doc = Document(page_content="Alice works at Microsoft.")
        result = transformer.process_response(doc)

        assert isinstance(result, GraphDocument)
        assert len(result.nodes) == 1
        assert result.nodes[0].id == "Alice"

    def test_process_response_sets_source_document(self):
        llm = self._make_mock_llm()
        transformer = LLMGraphTransformer(llm=llm)

        parsed_mock = MagicMock()
        parsed_mock.nodes = []
        parsed_mock.relationships = []
        transformer.chain = MagicMock()
        transformer.chain.invoke.return_value = {"parsed": parsed_mock, "raw": None}

        doc = Document(page_content="Some content.")
        result = transformer.process_response(doc)

        assert result.source == doc

    def test_strict_mode_filters_disallowed_nodes(self):
        llm = self._make_mock_llm()
        transformer = LLMGraphTransformer(
            llm=llm,
            allowed_nodes=["Person"],
            strict_mode=True,
        )

        allowed_node = MagicMock()
        allowed_node.id = "Alice"
        allowed_node.type = "Person"
        allowed_node.properties = None

        disallowed_node = MagicMock()
        disallowed_node.id = "Microsoft"
        disallowed_node.type = "Company"   # not in allowed_nodes
        disallowed_node.properties = None

        parsed_mock = MagicMock()
        parsed_mock.nodes = [allowed_node, disallowed_node]
        parsed_mock.relationships = []

        transformer.chain = MagicMock()
        transformer.chain.invoke.return_value = {"parsed": parsed_mock, "raw": None}

        doc = Document(page_content="Alice works at Microsoft.")
        result = transformer.process_response(doc)

        node_types = {n.type for n in result.nodes}
        assert "Company" not in node_types
