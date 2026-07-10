"""
Unit tests for GarageMind M1 - Report generator.
Covers health scoring, repair planning, freeze-frame correlation
(including the zero-variance edge case), and report structure.
Run with: pytest
"""

import pytest

from src.scan_engine.report_generator import (
    assess_health, build_repair_plan, correlate_freeze_frames,
    generate_report, _effort_for,
)


# ---------- Health assessment ----------

class TestHealthAssessment:
    def test_no_faults_is_perfect(self):
        health = assess_health({"results": []})
        assert health["score"] == 100
        assert health["action_required"] is False
        assert health["critical_count"] == 0

    def test_critical_fault_triggers_action(self):
        interp = {"results": [{"severity_key": "high"}]}
        health = assess_health(interp)
        assert health["critical_count"] == 1
        assert health["action_required"] is True
        assert health["score"] < 100

    def test_score_decreases_with_faults(self):
        one = assess_health({"results": [{"severity_key": "medium"}]})
        many = assess_health({"results": [{"severity_key": "high"},
                                          {"severity_key": "high"},
                                          {"severity_key": "medium"}]})
        assert many["score"] < one["score"]

    def test_score_never_negative(self):
        interp = {"results": [{"severity_key": "high"}] * 20}
        health = assess_health(interp)
        assert health["score"] >= 0


# ---------- Repair plan ----------

class TestRepairPlan:
    def test_priority_orders_by_severity(self):
        faults = [
            {"code": "P0401", "severity_key": "medium", "description_fr": "EGR",
             "likely_causes_fr": ["vanne egr"]},
            {"code": "P0301", "severity_key": "high", "description_fr": "cylindre 1",
             "likely_causes_fr": ["bougie"]},
        ]
        plan = build_repair_plan(faults)
        assert plan[0]["code"] == "P0301"  # high severity first
        assert plan[0]["priority"] == 1

    def test_effort_lookup_known_subsystem(self):
        fault = {"description_fr": "filtre a particules FAP",
                 "likely_causes_fr": ["fap colmate"]}
        effort = _effort_for(fault)
        assert effort["effort"] == "eleve"
        assert effort["hours_min"] is not None

    def test_effort_unknown_subsystem(self):
        fault = {"description_fr": "defaut inconnu xyz", "likely_causes_fr": []}
        effort = _effort_for(fault)
        assert effort["effort"] == "a evaluer"
        assert effort["hours_min"] is None


# ---------- Freeze-frame correlation ----------

class TestCorrelation:
    def test_value_inside_range_is_ok(self):
        faults = [{"code": "P0301",
                   "freeze_frame": {"engine_rpm": {"value": 617, "unit": "rpm"}}}]
        stats = {"N": {"mean": 617, "std": 15, "min": 596, "max": 662,
                       "latest": 650, "count": 100}}
        corr = correlate_freeze_frames(faults, stats)
        chk = corr[0]["quantitative_checks"][0]
        assert chk["within_live_range"] is True
        assert chk["z_score"] is not None

    def test_value_outside_range_is_atypical(self):
        faults = [{"code": "P0401",
                   "freeze_frame": {"engine_rpm": {"value": 1850, "unit": "rpm"}}}]
        stats = {"N": {"mean": 617, "std": 15, "min": 596, "max": 662,
                       "latest": 650, "count": 100}}
        corr = correlate_freeze_frames(faults, stats)
        chk = corr[0]["quantitative_checks"][0]
        assert chk["within_live_range"] is False

    def test_zero_variance_no_zscore(self):
        """The critical edge case: constant signal must not produce a z-score."""
        faults = [{"code": "P0401",
                   "freeze_frame": {"vehicle_speed": {"value": 45, "unit": "km/h"}}}]
        stats = {"VS": {"mean": 0, "std": 0, "min": 0, "max": 0,
                        "latest": 0, "count": 100}}
        corr = correlate_freeze_frames(faults, stats)
        chk = corr[0]["quantitative_checks"][0]
        assert chk["z_score"] is None
        assert chk["z_note"] is not None
        assert chk["within_live_range"] is False

    def test_context_notes_generated(self):
        faults = [{"code": "P0301",
                   "freeze_frame": {"engine_rpm": {"value": 600, "unit": "rpm"},
                                    "vehicle_speed": {"value": 0, "unit": "km/h"}}}]
        corr = correlate_freeze_frames(faults, {})
        notes = corr[0]["context_notes_fr"]
        assert any("ralenti" in n for n in notes)
        assert any("arret" in n for n in notes)


# ---------- Full report structure ----------

class TestFullReport:
    def test_report_has_all_sections(self):
        report = generate_report(lang="fr")
        for section in ["report_metadata", "vehicle_identity", "health_assessment",
                        "diagnostic_scan", "context_correlation", "repair_plan",
                        "transport_stats"]:
            assert section in report

    def test_report_has_disclaimer(self):
        report = generate_report(lang="fr")
        assert "disclaimer" in report["report_metadata"]

    def test_demo_vehicle_is_unhealthy(self):
        """The demo ECU has a critical misfire, so health must flag action."""
        report = generate_report(lang="fr")
        assert report["health_assessment"]["action_required"] is True
        assert report["diagnostic_scan"]["dtc_count"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])