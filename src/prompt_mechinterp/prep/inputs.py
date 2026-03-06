#!/usr/bin/env python3
"""Assemble test_cases.json from prompt text, region config, and conversations.

Usage:
    python -m prompt_mechinterp.prep.inputs \\
        --prompt system_prompt.txt \\
        --regions regions.json \\
        --conversations conversations.json \\
        --output test_cases.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from .regions import annotate_text, load_region_config, parse_query_positions, parse_tracked_tokens


def build_test_cases(
    system_prompt: str,
    conversations: List[dict],
    region_config: dict,
    num_samples: int = 0,
) -> dict:
    """Build test_cases.json structure from inputs.

    Args:
        system_prompt: Full system prompt text.
        conversations: List of conversation dicts, each with:
            - id: case identifier
            - user_message: user turn text
            - response: assistant response text
            - (optional) user_regions: list of region defs for user message
            - (optional) response_regions: list of region defs for response
        region_config: Loaded region configuration dict.
        num_samples: Max samples to include (0 = all).

    Returns:
        Dict ready to serialize as test_cases.json.
    """
    # Annotate system prompt regions
    sys_region_defs = region_config.get("system_prompt", {}).get("regions", [])
    system_regions = annotate_text(system_prompt, sys_region_defs)

    # Build cases
    cases = []
    user_region_defs = region_config.get("user_message", {}).get("regions", [])

    for conv in conversations:
        if num_samples > 0 and len(cases) >= num_samples:
            break

        user_msg = conv["user_message"]
        response = conv.get("response", "")

        # Annotate user message regions
        user_regions = annotate_text(user_msg, user_region_defs)

        # Annotate response regions if config provides them
        resp_region_defs = region_config.get("response", {}).get("regions", [])
        response_regions = annotate_text(response, resp_region_defs)

        # Override with per-conversation region defs if provided
        if "user_regions" in conv:
            user_regions = annotate_text(user_msg, conv["user_regions"])
        if "response_regions" in conv:
            response_regions = annotate_text(response, conv["response_regions"])

        case = {
            "id": conv["id"],
            "user_message": user_msg,
            "response": response,
            "user_regions": user_regions,
            "response_regions": response_regions,
        }
        cases.append(case)

    # Extract query positions and tracked tokens from config
    query_positions = parse_query_positions(region_config)
    tracked_tokens = parse_tracked_tokens(region_config)

    result = {
        "system_prompt": system_prompt,
        "system_regions": system_regions,
        "query_positions": query_positions,
        "tracked_tokens": tracked_tokens,
        "cases": cases,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Assemble test_cases.json for MI analysis",
    )
    parser.add_argument(
        "--prompt", required=True,
        help="Path to system prompt text file",
    )
    parser.add_argument(
        "--regions", required=True,
        help="Path to regions JSON config",
    )
    parser.add_argument(
        "--conversations", required=True,
        help="Path to conversations JSON (array of {id, user_message, response})",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output test_cases.json path",
    )
    parser.add_argument(
        "--samples", type=int, default=0,
        help="Max number of samples (0 = all)",
    )
    args = parser.parse_args()

    # Load inputs
    system_prompt = Path(args.prompt).read_text()
    region_config = load_region_config(args.regions)

    with open(args.conversations) as f:
        conversations = json.load(f)

    print(f"System prompt: {len(system_prompt)} chars")
    print(f"Conversations: {len(conversations)}")

    # Build test cases
    test_cases = build_test_cases(
        system_prompt, conversations, region_config, args.samples
    )

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(test_cases, f, indent=2, ensure_ascii=False)

    print(f"Written: {output_path}")
    print(f"  System regions: {len(test_cases['system_regions'])}")
    print(f"  Cases: {len(test_cases['cases'])}")
    if test_cases["tracked_tokens"]:
        print(f"  Tracked tokens: {test_cases['tracked_tokens']}")


if __name__ == "__main__":
    main()
