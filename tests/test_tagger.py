"""Tests for Tagger class and phrase extraction."""

import pytest

from dp_fusion_lib import find_phrase_offsets


class TestFindPhraseOffsets:
    """Tests for find_phrase_offsets function."""

    def test_single_phrase(self):
        """Test finding a single phrase."""
        text = "My name is John Smith and I live here."
        phrases = ["John Smith"]

        offsets = find_phrase_offsets(text, phrases)

        assert len(offsets) == 1
        assert text[offsets[0][0]:offsets[0][1]] == "John Smith"

    def test_multiple_phrases(self):
        """Test finding multiple different phrases."""
        text = "John Smith visited New York on Monday."
        phrases = ["John Smith", "New York", "Monday"]

        offsets = find_phrase_offsets(text, phrases)

        assert len(offsets) == 3

        found_texts = [text[o[0]:o[1]] for o in offsets]
        assert "John Smith" in found_texts
        assert "New York" in found_texts
        assert "Monday" in found_texts

    def test_repeated_phrase(self):
        """Test finding a phrase that appears multiple times."""
        text = "John met John at John's house."
        phrases = ["John"]

        offsets = find_phrase_offsets(text, phrases)

        assert len(offsets) == 3
        for offset in offsets:
            assert text[offset[0]:offset[1]] == "John"

    def test_no_match(self):
        """Test when phrase is not found."""
        text = "Hello world."
        phrases = ["John Smith"]

        offsets = find_phrase_offsets(text, phrases)

        assert len(offsets) == 0

    def test_empty_phrases(self):
        """Test with empty phrase list."""
        text = "Some text here."
        phrases = []

        offsets = find_phrase_offsets(text, phrases)

        assert len(offsets) == 0

    def test_overlapping_matches(self):
        """Test phrases that overlap in the text."""
        text = "New York City is great."
        phrases = ["New York", "York City"]

        offsets = find_phrase_offsets(text, phrases)

        # Both should be found
        assert len(offsets) == 2

    def test_offset_values(self):
        """Test that offset values are correct."""
        text = "Hello John!"
        phrases = ["John"]

        offsets = find_phrase_offsets(text, phrases)

        assert len(offsets) == 1
        assert offsets[0][0] == 6  # "John" starts at index 6
        assert offsets[0][1] == 10  # "John" ends at index 10

    def test_case_sensitive(self):
        """Test that matching is case-sensitive."""
        text = "John and john are different."
        phrases = ["John"]

        offsets = find_phrase_offsets(text, phrases)

        assert len(offsets) == 1
        assert text[offsets[0][0]:offsets[0][1]] == "John"
