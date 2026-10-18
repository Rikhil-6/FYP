import pytest

from fyp_drift.evaluation.metrics import classwise_model_comparison


def _row(truth, old, new, routed, routed_to_new):
    return {
        "true_class": truth,
        "old_class": old,
        "new_class": new,
        "routed_class": routed,
        "routed_to_new": routed_to_new,
    }


def test_classwise_comparison_reports_support_and_model_deltas():
    records = [
        _row(0, 0, 0, 0, False),
        _row(0, 1, 0, 0, True),
        _row(1, 1, 1, 1, False),
        _row(1, 1, 0, 1, False),
        _row(2, 2, 2, 2, False),
    ]

    report = classwise_model_comparison(records, classes=3, min_support=2)

    assert report["eligible_classes"] == 2
    assert report["excluded_low_support_classes"] == 1
    assert report["comparison"]["new"]["classes_improved"] == 1
    assert report["comparison"]["new"]["classes_worsened"] == 1
    # Correcting class 0 also removes a false positive from class 1.
    assert report["comparison"]["routed"]["classes_improved"] == 2
    assert report["comparison"]["routed"]["classes_unchanged"] == 0
    assert report["classes"][0]["routing"]["helpful"] == 1
    assert report["classes"][0]["routing"]["harmful"] == 0
    assert report["classes"][0]["routing"]["beneficial_available"] == 1
    assert report["classes"][0]["routing"]["beneficial_missed"] == 0


def test_classwise_comparison_does_not_claim_unsupported_aggregate_gains():
    report = classwise_model_comparison(
        [_row(0, 0, 0, 0, False)], classes=2, min_support=2
    )

    assert report["eligible_classes"] == 0
    assert report["historical_f1_rank_strata"] == []
    assert report["comparison"]["new"]["mean_f1_delta_pp"] is None


def test_classwise_comparison_rejects_invalid_support_threshold():
    with pytest.raises(ValueError, match="at least 1"):
        classwise_model_comparison([_row(0, 0, 0, 0, False)], classes=1, min_support=0)
