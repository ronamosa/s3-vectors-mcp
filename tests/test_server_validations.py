import json

import pytest

from s3_vectors_mcp import server


def test_validate_text_field_accepts_non_empty():
    server.validate_text_field("hello", "text")


def test_validate_text_field_rejects_blank():
    with pytest.raises(ValueError):
        server.validate_text_field("   ", "text")


@pytest.mark.parametrize("value", [0, -1, 1001])
def test_validate_top_k_bounds(value):
    with pytest.raises(ValueError):
        server.validate_top_k(value)


def test_serialize_filter_expression_roundtrip():
    filter_expr = {"category": {"$eq": "docs"}}
    serialized = server.serialize_filter_expression(filter_expr)
    assert isinstance(serialized, str)
    assert json.loads(serialized) == filter_expr

