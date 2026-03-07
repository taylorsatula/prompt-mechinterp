"""Tests for dynamic phase scaling in constants.py."""

from prompt_mechinterp.constants import analysis_phases, display_phases


class TestDisplayPhases:
    def test_64_layers_produces_four_phases(self):
        phases = display_phases(64)
        assert len(phases) == 4

    def test_64_layers_spans_full_range(self):
        phases = display_phases(64)
        assert phases[0][1] == 0, "First phase should start at layer 0"
        assert phases[-1][2] == 63, "Last phase should end at last layer"

    def test_32_layers_phases_scale_proportionally(self):
        phases_64 = display_phases(64)
        phases_32 = display_phases(32)

        # Each phase should cover roughly half the layer range compared to 64
        for (_, s64, e64), (_, s32, e32) in zip(phases_64, phases_32):
            span_64 = e64 - s64
            span_32 = e32 - s32
            # Allow rounding tolerance of 1
            assert abs(span_32 - span_64 / 2) <= 1

    def test_no_gaps_between_phases(self):
        """Adjacent phases should have no gap: next start <= prev end + 1."""
        for n in (32, 64, 80, 128):
            phases = display_phases(n)
            for i in range(len(phases) - 1):
                _, _, end = phases[i]
                _, start_next, _ = phases[i + 1]
                assert start_next <= end + 1, (
                    f"Gap between phases {i} and {i+1} for {n} layers: "
                    f"end={end}, next_start={start_next}"
                )

    def test_no_overlaps_between_phases(self):
        """Each phase should start at or after the previous phase ends."""
        for n in (32, 64, 80, 128):
            phases = display_phases(n)
            for i in range(1, len(phases)):
                _, start, _ = phases[i]
                _, _, prev_end = phases[i - 1]
                assert start >= prev_end, (
                    f"Phase {i} starts at {start} but phase {i-1} "
                    f"ends at {prev_end} for {n} layers"
                )

    def test_phases_have_labels(self):
        phases = display_phases(64)
        for label, _, _ in phases:
            assert isinstance(label, str) and len(label) > 0


class TestAnalysisPhases:
    def test_64_layers_produces_five_named_phases(self):
        phases = analysis_phases(64)
        assert len(phases) == 5

    def test_all_phases_within_layer_range(self):
        for n in (32, 64, 80):
            phases = analysis_phases(n)
            for name, (start, end) in phases.items():
                assert 0 <= start <= n - 1, f"{name} start={start} out of [0, {n-1}]"
                assert 0 <= end <= n - 1, f"{name} end={end} out of [0, {n-1}]"

    def test_first_phase_starts_at_zero(self):
        phases = analysis_phases(64)
        first_start = min(start for start, _ in phases.values())
        assert first_start == 0

    def test_last_phase_ends_at_final_layer(self):
        phases = analysis_phases(64)
        last_end = max(end for _, end in phases.values())
        assert last_end == 63

    def test_phases_scale_to_different_layer_counts(self):
        phases_64 = analysis_phases(64)
        phases_32 = analysis_phases(32)
        # Same phase names
        assert set(phases_64.keys()) == set(phases_32.keys())
        # Last phase end should match last layer
        last_end_32 = max(end for _, end in phases_32.values())
        assert last_end_32 == 31
