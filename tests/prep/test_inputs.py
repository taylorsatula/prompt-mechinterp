"""Tests for prep/inputs.py — test case assembly."""

from prompt_mechinterp.prep.inputs import build_test_cases


def _basic_region_config():
    return {
        "system_prompt": {
            "regions": [
                {"name": "rules", "start_marker": "## Rules", "end_marker": "## Examples"},
                {"name": "examples", "start_marker": "## Examples"},
            ],
        },
        "user_message": {
            "regions": [
                {"name": "current_message", "start_marker": "Current:"},
            ],
        },
        "query_positions": {"terminal": "last_token"},
        "tracked_tokens": ["<"],
    }


def _basic_conversations():
    return [
        {
            "id": "case_01",
            "user_message": "Previous: context\nCurrent: question",
            "response": "The answer is 42",
        },
        {
            "id": "case_02",
            "user_message": "Previous: other\nCurrent: another question",
            "response": "Response here",
        },
    ]


class TestBuildTestCases:
    def test_basic_assembly(self):
        prompt = "Intro\n## Rules\nDo things\n## Examples\nSee examples"
        config = _basic_region_config()
        convs = _basic_conversations()

        result = build_test_cases(prompt, convs, config)

        assert result["system_prompt"] == prompt
        assert "rules" in result["system_regions"]
        assert "examples" in result["system_regions"]
        assert len(result["cases"]) == 2
        assert result["query_positions"] == {"terminal": "last_token"}
        assert result["tracked_tokens"] == ["<"]

    def test_user_regions_have_correct_boundaries(self):
        prompt = "## Rules\nStuff\n## Examples\nMore"
        config = _basic_region_config()
        convs = _basic_conversations()

        result = build_test_cases(prompt, convs, config)
        case = result["cases"][0]

        assert "current_message" in case["user_regions"]
        # "Current:" starts at position 19 in "Previous: context\nCurrent: question"
        user_msg = case["user_message"]
        expected_start = user_msg.index("Current:")
        assert case["user_regions"]["current_message"]["char_start"] == expected_start

    def test_per_conversation_region_override(self):
        prompt = "## Rules\nStuff\n## Examples\nMore"
        config = _basic_region_config()
        convs = [
            {
                "id": "case_override",
                "user_message": "Custom: data here",
                "response": "ok",
                "user_regions": [
                    {"name": "custom_region", "start_marker": "Custom:"},
                ],
            }
        ]

        result = build_test_cases(prompt, convs, config)
        case = result["cases"][0]
        assert "custom_region" in case["user_regions"]
        # Default user_message regions should NOT be present
        assert "current_message" not in case["user_regions"]

    def test_sample_limiting(self):
        prompt = "## Rules\nStuff\n## Examples\nMore"
        config = _basic_region_config()
        convs = [
            {"id": f"case_{i}", "user_message": "Current: hi", "response": "ok"}
            for i in range(5)
        ]

        result = build_test_cases(prompt, convs, config, num_samples=2)
        assert len(result["cases"]) == 2

    def test_optional_response_defaults_empty(self):
        prompt = "## Rules\nStuff\n## Examples\nMore"
        config = _basic_region_config()
        convs = [
            {"id": "no_response", "user_message": "Current: question"},
        ]

        result = build_test_cases(prompt, convs, config)
        assert result["cases"][0]["response"] == ""

    def test_num_samples_zero_means_all(self):
        prompt = "## Rules\n\n## Examples\n"
        config = _basic_region_config()
        convs = [
            {"id": f"c{i}", "user_message": "Current: x", "response": "y"}
            for i in range(10)
        ]

        result = build_test_cases(prompt, convs, config, num_samples=0)
        assert len(result["cases"]) == 10

    def test_empty_conversations(self):
        prompt = "## Rules\n\n## Examples\n"
        config = _basic_region_config()

        result = build_test_cases(prompt, [], config)
        assert result["cases"] == []
        assert "system_regions" in result
