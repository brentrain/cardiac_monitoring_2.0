"""Human review store: rhythm confirmations, corrections, and event verdicts."""
from icu_telemetry_server import (
    EVENT_CONFIRMED_CLINICAL, EVENT_FALSE_ALARM, EVENT_PENDING, HumanInLoopStore,
)


def test_confirm_rhythm_records_agreement():
    store = HumanInLoopStore()
    store.confirm_algorithm_rhythm(3)
    assert store.adjudication_for_patient(3)["rhythm_agrees"] is True


def test_correct_rhythm_with_valid_label_is_stored():
    store = HumanInLoopStore()
    assert store.correct_rhythm(3, "Atrial Fibrillation") is True
    review = store.adjudication_for_patient(3)
    assert review["rhythm_agrees"] is False
    assert review["human_rhythm"] == "Atrial Fibrillation"


def test_correct_rhythm_rejects_unknown_label():
    store = HumanInLoopStore()
    assert store.correct_rhythm(3, "Not a real rhythm") is False
    assert store.adjudication_for_patient(3) == {}


def test_clear_patient_removes_review():
    store = HumanInLoopStore()
    store.confirm_algorithm_rhythm(3)
    store.clear_patient(3)
    assert store.adjudication_for_patient(3) == {}


def test_event_verdict_defaults_to_pending():
    store = HumanInLoopStore()
    assert store.event_verdict(10) == EVENT_PENDING


def test_event_verdict_can_be_set_and_reset():
    store = HumanInLoopStore()
    assert store.set_event_verdict(10, EVENT_CONFIRMED_CLINICAL) is True
    assert store.event_verdict(10) == EVENT_CONFIRMED_CLINICAL
    store.set_event_verdict(10, EVENT_PENDING)
    assert store.event_verdict(10) == EVENT_PENDING


def test_event_verdict_rejects_unknown_value():
    store = HumanInLoopStore()
    assert store.set_event_verdict(10, "not_a_verdict") is False
    assert store.event_verdict(10) == EVENT_PENDING


def test_false_alarm_verdict_is_kept():
    store = HumanInLoopStore()
    store.set_event_verdict(7, EVENT_FALSE_ALARM)
    assert store.event_verdict(7) == EVENT_FALSE_ALARM
