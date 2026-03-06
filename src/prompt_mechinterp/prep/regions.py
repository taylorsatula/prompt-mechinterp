"""Region annotation engine — map named text spans to character offsets.

The user defines regions via a JSON config file. This module finds each
region's character boundaries in the assembled prompt text, producing
the character-level region map that downstream code (run_analysis.py)
converts to token-level via cumulative decode mapping.

Region config format (regions.json):
    {
      "system_prompt": {
        "regions": [
          {"name": "rules", "start_marker": "## Rules", "end_marker": "## Examples"},
          {"name": "examples", "start_marker": "## Examples", "end_marker": null}
        ]
      },
      "user_message": {
        "regions": [
          {"name": "context", "start_marker": "Previous:", "end_marker": "Current:"},
          {"name": "current", "start_marker": "Current:", "end_marker": null}
        ]
      },
      "query_positions": {
        "terminal": "last_token",
        "decision": {"after_text": "Folder:"}
      },
      "tracked_tokens": ["<", "folder_a"]
    }

Detection strategies:
    - Marker-based: start_marker / end_marker strings (literal text boundaries)
    - Regex-based: start_pattern / end_pattern (for complex boundaries)
    - Character range: start_char / end_char (explicit offsets)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_region_config(path: str) -> dict:
    """Load region configuration from a JSON file."""
    with open(path) as f:
        return json.load(f)


def annotate_text(
    text: str,
    region_defs: List[dict],
    text_offset: int = 0,
) -> Dict[str, Dict[str, int]]:
    """Find character-level boundaries for each region definition in text.

    Args:
        text: The text to annotate.
        region_defs: List of region definition dicts, each with 'name' and
            boundary specifiers (start_marker/end_marker, start_pattern/end_pattern,
            or start_char/end_char).
        text_offset: Global character offset to add to all positions (for when
            text is a substring of a larger assembled prompt).

    Returns:
        Dict mapping region_name -> {"char_start": int, "char_end": int}
    """
    regions = {}

    for defn in region_defs:
        name = defn["name"]
        start, end = _find_boundaries(text, defn)

        if start is None:
            print(f"  WARNING: Region '{name}' start not found in text")
            continue

        if end is None:
            end = len(text)

        regions[name] = {
            "char_start": start + text_offset,
            "char_end": end + text_offset,
        }

        # Handle nested sub-regions
        if "regions" in defn:
            sub_text = text[start:end]
            sub_regions = annotate_text(
                sub_text,
                defn["regions"],
                text_offset=start + text_offset,
            )
            regions.update(sub_regions)

    return regions


def _find_boundaries(
    text: str, defn: dict
) -> Tuple[Optional[int], Optional[int]]:
    """Find start and end positions for a region definition."""

    # Strategy 1: Explicit character offsets
    if "start_char" in defn:
        start = defn["start_char"]
        end = defn.get("end_char", None)
        return start, end

    # Strategy 2: Regex patterns
    if "start_pattern" in defn:
        m = re.search(defn["start_pattern"], text)
        if m is None:
            return None, None
        start = m.start()

        end = None
        if "end_pattern" in defn and defn["end_pattern"]:
            m2 = re.search(defn["end_pattern"], text[m.end() :])
            if m2:
                end = m.end() + m2.start()
        return start, end

    # Strategy 3: Literal marker strings (most common)
    if "start_marker" in defn:
        start_marker = defn["start_marker"]
        idx = text.find(start_marker)
        if idx == -1:
            return None, None
        start = idx

        end = None
        end_marker = defn.get("end_marker")
        if end_marker:
            end_idx = text.find(end_marker, start + len(start_marker))
            if end_idx != -1:
                end = end_idx
        return start, end

    return None, None


def parse_query_positions(
    config: dict,
) -> dict:
    """Extract query position definitions from region config.

    Returns dict suitable for inclusion in test_cases.json.
    """
    return config.get("query_positions", {})


def parse_tracked_tokens(config: dict) -> List[str]:
    """Extract tracked token list from region config."""
    return config.get("tracked_tokens", [])
