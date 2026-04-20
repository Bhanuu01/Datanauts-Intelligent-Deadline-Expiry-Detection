"""
Unit tests for predict.py helpers.

Run with:  pytest tests/test_predict.py -v

Note: predict() itself requires real model paths and is tested via the
end-to-end evaluate.py pipeline.  These tests cover the pure-Python helpers
that are independently verifiable.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from predict import _resolve_date, _extract_entities, _get_pipelines, _PIPELINE_CACHE


# ── _resolve_date ─────────────────────────────────────────────────────────────

class TestResolveDate:
    def test_iso_passthrough(self):
        assert _resolve_date("2025-12-31") == "2025-12-31"

    def test_written_date(self):
        assert _resolve_date("December 31, 2025") == "2025-12-31"

    def test_slash_date(self):
        assert _resolve_date("12/31/2025") == "2025-12-31"

    def test_written_month_year(self):
        result = _resolve_date("January 2024")
        assert result is not None
        assert result.startswith("2024-01")

    def test_invalid_returns_none(self):
        assert _resolve_date("not a date") is None
        assert _resolve_date("") is None
        assert _resolve_date(None) is None

    def test_ambiguous_prefers_mdy(self):
        result = _resolve_date("01/02/2025")
        assert result == "2025-01-02"


# ── _extract_entities ─────────────────────────────────────────────────────────

def _make_tok(entity, word, start=0):
    return {"entity": entity, "word": word, "start": start, "end": start + len(word), "score": 0.99}


class TestExtractEntities:
    def test_single_b_tag(self):
        ner_out = [_make_tok("B-EXP_DATE", "December")]
        result  = _extract_entities(ner_out, {"EXP_DATE"})
        assert len(result) == 1
        assert result[0]["entity_type"] == "EXP_DATE"
        assert result[0]["text"] == "December"

    def test_bio_span_joined(self):
        ner_out = [
            _make_tok("B-EXP_DATE", "December", 0),
            _make_tok("I-EXP_DATE", "31",       9),
            _make_tok("I-EXP_DATE", "2025",     12),
        ]
        result = _extract_entities(ner_out, {"EXP_DATE"})
        assert len(result) == 1
        assert "December" in result[0]["text"]
        assert "2025" in result[0]["text"]

    def test_subword_hash_concatenated(self):
        ner_out = [
            _make_tok("B-EXP_DATE", "Decem",  0),
            _make_tok("I-EXP_DATE", "##ber", 5),
        ]
        result = _extract_entities(ner_out, {"EXP_DATE"})
        assert len(result) == 1
        assert result[0]["text"] == "December"

    def test_not_in_allowed_types_skipped(self):
        ner_out = [_make_tok("B-EXP_DATE", "December")]
        result  = _extract_entities(ner_out, {"START_DATE"})
        assert result == []

    def test_multiple_spans(self):
        ner_out = [
            _make_tok("B-EXP_DATE",   "January",  0),
            _make_tok("O",            "and",      8),
            _make_tok("B-START_DATE", "February", 12),
        ]
        result = _extract_entities(ner_out, {"EXP_DATE", "START_DATE"})
        assert len(result) == 2
        types  = {r["entity_type"] for r in result}
        assert types == {"EXP_DATE", "START_DATE"}

    def test_empty_ner_output(self):
        assert _extract_entities([], {"EXP_DATE"}) == []

    def test_only_o_tags(self):
        ner_out = [_make_tok("O", "The"), _make_tok("O", "contract")]
        assert _extract_entities(ner_out, {"EXP_DATE"}) == []

    def test_i_tag_without_b_tag_ignored(self):
        ner_out = [_make_tok("I-EXP_DATE", "31")]
        result  = _extract_entities(ner_out, {"EXP_DATE"})
        assert result == []


# ── _get_pipelines / cache ────────────────────────────────────────────────────

class TestPipelineCache:
    def test_cache_populated_on_first_call(self, monkeypatch):
        sentinel = object()
        calls    = []

        def mock_pipeline(*args, **kwargs):
            calls.append(kwargs.get("model"))
            return sentinel

        import predict as pred_mod
        monkeypatch.setattr(pred_mod, "pipeline", mock_pipeline)
        pred_mod._PIPELINE_CACHE.clear()

        p1, p2 = pred_mod._get_pipelines("clf_model", "ner_model")
        assert p1 is sentinel
        assert p2 is sentinel
        assert len(calls) == 2

    def test_cache_reused_on_second_call(self, monkeypatch):
        calls = []

        def mock_pipeline(*args, **kwargs):
            calls.append(1)
            return object()

        import predict as pred_mod
        monkeypatch.setattr(pred_mod, "pipeline", mock_pipeline)
        pred_mod._PIPELINE_CACHE.clear()

        pred_mod._get_pipelines("clf_a", "ner_a")
        first_count = len(calls)
        pred_mod._get_pipelines("clf_a", "ner_a")
        assert len(calls) == first_count, "pipeline() called again — cache not working"

    def test_different_keys_load_separate_models(self, monkeypatch):
        loaded = []

        def mock_pipeline(*args, **kwargs):
            loaded.append(kwargs.get("model"))
            return object()

        import predict as pred_mod
        monkeypatch.setattr(pred_mod, "pipeline", mock_pipeline)
        pred_mod._PIPELINE_CACHE.clear()

        pred_mod._get_pipelines("clf_v1", "ner_v1")
        pred_mod._get_pipelines("clf_v2", "ner_v2")
        assert len(loaded) == 4, "Expected 4 pipeline loads for 2 distinct model pairs"
