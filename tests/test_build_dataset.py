"""
Unit tests for build_dataset.py helpers.

Run with:  pytest tests/test_build_dataset.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from build_dataset import normalise, sent_split, bio_tag_regex, assert_no_split_leakage


# ── normalise ────────────────────────────────────────────────────────────────

class TestNormalise:
    def test_strips_whitespace(self):
        assert normalise("  hello world  ") == "hello world"

    def test_collapses_internal_spaces(self):
        assert normalise("hello   world") == "hello world"

    def test_empty_string(self):
        assert normalise("") == ""

    def test_none_returns_empty(self):
        assert normalise(None) == ""

    def test_removes_non_ascii(self):
        result = normalise("caf\u00e9")
        assert result.isascii()


# ── sent_split ───────────────────────────────────────────────────────────────

class TestSentSplit:
    def test_basic_split(self):
        text = (
            "This agreement shall expire on December 31, 2025. "
            "The effective date shall be January 1, 2024."
        )
        sents = sent_split(text)
        assert len(sents) == 2

    def test_empty_input(self):
        assert sent_split("") == []
        assert sent_split(None) == []

    def test_short_fragments_merged(self):
        text = "This agreement will terminate on December 31, 2025. See above."
        sents = sent_split(text)
        assert all(len(s.split()) >= 8 for s in sents)

    def test_abbreviation_not_split(self):
        text = (
            "The company Inc. will renew the contract for three years. "
            "The expiry date is Dec. 31, 2025."
        )
        sents = sent_split(text)
        assert len(sents) == 2

    def test_clause_number_stripped(self):
        text = "1. This Agreement shall commence on the Effective Date. 2. The term is one year."
        sents = sent_split(text)
        for s in sents:
            assert not s[0].isdigit()

    def test_long_sentence_split_on_semicolon(self):
        base = "word " * 10
        long_text = (base + "; " + base) * 5
        sents = sent_split(long_text)
        assert all(len(s) <= 510 for s in sents)

    def test_newlines_normalised(self):
        text = "This agreement\nshall expire on December 31, 2025.\nSee above for more details."
        sents = sent_split(text)
        assert all("\n" not in s for s in sents)


# ── bio_tag_regex ─────────────────────────────────────────────────────────────

class TestBioTagRegex:
    def _tokens(self, text):
        return text.split()

    def test_none_type_all_O(self):
        tokens = self._tokens("This is a sentence with no date information here please")
        tags = bio_tag_regex(tokens, "none")
        assert all(t == "O" for t in tags)
        assert len(tags) == len(tokens)

    def test_expiration_date_tagged(self):
        tokens = self._tokens("The contract expires on December 31 2025 unless renewed")
        tags = bio_tag_regex(tokens, "expiration")
        assert "B-EXP_DATE" in tags

    def test_effective_date_tagged(self):
        tokens = self._tokens("This agreement is effective as of January 1 2024 and beyond")
        tags = bio_tag_regex(tokens, "effective")
        assert "B-START_DATE" in tags

    def test_agreement_date_tagged(self):
        tokens = self._tokens("This agreement is dated as of March 15 2023 between parties")
        tags = bio_tag_regex(tokens, "agreement")
        assert "B-START_DATE" in tags

    def test_renewal_duration_tagged(self):
        tokens = self._tokens("The contract shall renew for a period of three years automatically")
        tags = bio_tag_regex(tokens, "renewal")
        assert "B-DURATION" in tags

    def test_notice_period_date_tagged(self):
        tokens = self._tokens("Written notice must be given by January 31 2025 to terminate")
        tags = bio_tag_regex(tokens, "notice_period")
        assert "B-NOTICE_DATE" in tags

    def test_notice_period_duration_fallback(self):
        tokens = self._tokens("Termination requires sixty days written notice before renewal date")
        tags = bio_tag_regex(tokens, "notice_period")
        assert "B-DURATION" in tags

    def test_tag_length_matches_tokens(self):
        for ctype in ("expiration", "effective", "renewal", "notice_period", "agreement", "none"):
            tokens = self._tokens("The agreement expires on December 31 2025 at midnight")
            tags = bio_tag_regex(tokens, ctype)
            assert len(tags) == len(tokens), f"Length mismatch for ctype={ctype}"

    def test_bio_sequence_valid(self):
        tokens = self._tokens("The contract expires on December 31 2025 unless renewed early")
        tags = bio_tag_regex(tokens, "expiration")
        prev = None
        for tag in tags:
            if tag.startswith("I-"):
                entity = tag[2:]
                assert prev in (f"B-{entity}", f"I-{entity}"), \
                    f"I-tag without preceding B-tag: prev={prev}, tag={tag}"
            prev = tag


# ── assert_no_split_leakage ───────────────────────────────────────────────────

class TestAssertNoSplitLeakage:
    def _make_fake_split(self, filenames):
        return [{"Filename": fn} for fn in filenames]

    def test_clean_splits_pass(self):
        raw = {
            "train": self._make_fake_split(["a.txt", "b.txt"]),
            "val":   self._make_fake_split(["c.txt"]),
            "test":  self._make_fake_split(["d.txt"]),
        }
        assert_no_split_leakage(raw)  # should not raise

    def test_duplicate_across_train_val_raises(self):
        raw = {
            "train": self._make_fake_split(["a.txt", "b.txt"]),
            "val":   self._make_fake_split(["b.txt"]),
            "test":  self._make_fake_split(["c.txt"]),
        }
        with pytest.raises(AssertionError, match="LEAKAGE"):
            assert_no_split_leakage(raw)

    def test_duplicate_across_train_test_raises(self):
        raw = {
            "train": self._make_fake_split(["a.txt", "b.txt"]),
            "val":   self._make_fake_split(["c.txt"]),
            "test":  self._make_fake_split(["a.txt"]),
        }
        with pytest.raises(AssertionError, match="LEAKAGE"):
            assert_no_split_leakage(raw)
