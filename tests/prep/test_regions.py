"""Tests for prep/regions.py — boundary detection engine."""

from prompt_mechinterp.prep.regions import annotate_text


# ===========================================================================
# Marker-based detection
# ===========================================================================

class TestMarkerBased:
    def test_finds_marker_boundaries(self):
        text = "Intro text\n## Rules\nDo this\n## Examples\nSee here"
        regions = annotate_text(text, [
            {"name": "rules", "start_marker": "## Rules", "end_marker": "## Examples"},
        ])
        assert "rules" in regions
        r = regions["rules"]
        assert r["char_start"] == text.index("## Rules")
        assert r["char_end"] == text.index("## Examples")

    def test_missing_start_marker_skips_region(self):
        text = "No markers here"
        regions = annotate_text(text, [
            {"name": "rules", "start_marker": "## Rules", "end_marker": "## End"},
        ])
        assert "rules" not in regions

    def test_missing_end_marker_extends_to_end(self):
        """If the end boundary can't be found, the region extends to the end of text."""
        text = "Start\n## Rules\nContent goes here"
        regions = annotate_text(text, [
            {"name": "rules", "start_marker": "## Rules", "end_marker": "## End"},
        ])
        assert regions["rules"]["char_end"] == len(text)

    def test_null_end_marker_extends_to_end(self):
        text = "Start\n## Rules\nContent goes here"
        regions = annotate_text(text, [
            {"name": "rules", "start_marker": "## Rules", "end_marker": None},
        ])
        assert regions["rules"]["char_end"] == len(text)

    def test_end_marker_empty_string_extends_to_end(self):
        """An empty string end_marker should be treated as 'no end marker'."""
        text = "Start\n## Rules\nContent goes here"
        regions = annotate_text(text, [
            {"name": "rules", "start_marker": "## Rules", "end_marker": ""},
        ])
        assert regions["rules"]["char_end"] == len(text)


# ===========================================================================
# Regex-based detection
# ===========================================================================

class TestRegexBased:
    def test_finds_regex_boundaries(self):
        text = "Header\nSection 42 content here\nSection 99 more"
        regions = annotate_text(text, [
            {
                "name": "sec42",
                "start_pattern": r"Section \d+",
                "end_pattern": r"Section \d+",
            },
        ])
        assert "sec42" in regions
        r = regions["sec42"]
        assert text[r["char_start"]:r["char_start"] + 10] == "Section 42"
        # End should be at the start of "Section 99"
        assert text[r["char_end"]:r["char_end"] + 10] == "Section 99"

    def test_no_regex_match_skips_region(self):
        text = "No sections here"
        regions = annotate_text(text, [
            {"name": "sec", "start_pattern": r"Section \d+"},
        ])
        assert "sec" not in regions

    def test_regex_no_end_pattern(self):
        text = "Prefix Section 1 content"
        regions = annotate_text(text, [
            {"name": "sec", "start_pattern": r"Section \d+"},
        ])
        assert regions["sec"]["char_end"] == len(text)


# ===========================================================================
# Char-range detection
# ===========================================================================

class TestCharRange:
    def test_exact_offsets(self):
        text = "0123456789" * 10
        regions = annotate_text(text, [
            {"name": "mid", "start_char": 10, "end_char": 50},
        ])
        assert regions["mid"]["char_start"] == 10
        assert regions["mid"]["char_end"] == 50

    def test_start_char_only(self):
        text = "0123456789"
        regions = annotate_text(text, [
            {"name": "tail", "start_char": 5},
        ])
        assert regions["tail"]["char_start"] == 5
        assert regions["tail"]["char_end"] == len(text)


# ===========================================================================
# Nested sub-regions
# ===========================================================================

class TestNestedRegions:
    def test_children_have_global_offsets(self):
        text = "HEADER\n## Section\nRule A\nRule B\nEND"
        regions = annotate_text(text, [
            {
                "name": "section",
                "start_marker": "## Section",
                "end_marker": "END",
                "regions": [
                    {"name": "rule_a", "start_marker": "Rule A", "end_marker": "Rule B"},
                ],
            },
        ])
        assert "section" in regions
        assert "rule_a" in regions
        # rule_a's char_start should be its global position in the full text
        assert regions["rule_a"]["char_start"] == text.index("Rule A")
        assert regions["rule_a"]["char_end"] == text.index("Rule B")

    def test_children_relative_to_parent_text(self):
        """Children only search within the parent's text range."""
        text = "Rule X early\n## Parent\nRule X inside\n## End"
        regions = annotate_text(text, [
            {
                "name": "parent",
                "start_marker": "## Parent",
                "end_marker": "## End",
                "regions": [
                    {"name": "child", "start_marker": "Rule X"},
                ],
            },
        ])
        # Child should find "Rule X inside", not the earlier "Rule X early"
        child_start = regions["child"]["char_start"]
        assert child_start == text.index("Rule X inside")


# ===========================================================================
# text_offset parameter
# ===========================================================================

class TestTextOffset:
    def test_offset_shifts_all_positions(self):
        text = "## Rules\nContent"
        regions = annotate_text(text, [
            {"name": "rules", "start_marker": "## Rules"},
        ], text_offset=100)
        assert regions["rules"]["char_start"] == 100 + text.index("## Rules")
        assert regions["rules"]["char_end"] == 100 + len(text)

    def test_nested_offset_accumulates(self):
        text = "## Parent\nChild marker here"
        regions = annotate_text(text, [
            {
                "name": "parent",
                "start_marker": "## Parent",
                "regions": [
                    {"name": "child", "start_marker": "Child marker"},
                ],
            },
        ], text_offset=200)
        assert regions["child"]["char_start"] == 200 + text.index("Child marker")


# ===========================================================================
# Multiple regions
# ===========================================================================

class TestMultipleRegions:
    def test_multiple_sequential_regions(self):
        text = "## Rules\nRule text\n## Examples\nExample text\n## End"
        regions = annotate_text(text, [
            {"name": "rules", "start_marker": "## Rules", "end_marker": "## Examples"},
            {"name": "examples", "start_marker": "## Examples", "end_marker": "## End"},
        ])
        assert "rules" in regions
        assert "examples" in regions
        # Rules end should equal examples start
        assert regions["rules"]["char_end"] == regions["examples"]["char_start"]
