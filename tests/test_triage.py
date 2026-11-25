"""Triage rules for the dashboard status (vitals, rhythm, and artifact)."""
from icu_telemetry_simulator import (
    CardiacRhythm, Patient, TechnicalArtifact, VitalSigns,
)
from icu_telemetry_server import _patient_status


def make_patient(hr=75, spo2=98, sbp=120, rhythm=CardiacRhythm.NORMAL_SINUS,
                 artifact=TechnicalArtifact.NONE):
    p = Patient(patient_id=1, name="Test", age=50, diagnosis="Test")
    p.vitals = VitalSigns(heart_rate=hr, systolic_bp=sbp, diastolic_bp=80,
                          o2_saturation=spo2, respiratory_rate=16, temperature=37.0)
    p.current_rhythm = rhythm
    p.current_artifact = artifact
    return p


def test_normal_vitals_are_ok():
    level, reasons = _patient_status(make_patient())
    assert level == "ok"
    assert reasons == []


def test_no_pulse_is_critical():
    level, _ = _patient_status(make_patient(hr=0))
    assert level == "critical"


def test_low_spo2_is_critical():
    level, reasons = _patient_status(make_patient(spo2=88))
    assert level == "critical"
    assert any("Hypoxemia" in r for r in reasons)


def test_borderline_spo2_is_warning():
    level, _ = _patient_status(make_patient(spo2=91))
    assert level == "warning"


def test_fast_heart_rate_is_warning():
    level, reasons = _patient_status(make_patient(hr=150))
    assert level == "warning"
    assert any("Tachycardia" in r for r in reasons)


def test_artifact_alone_is_warning_not_ok():
    level, reasons = _patient_status(make_patient(artifact=TechnicalArtifact.LOOSE_LEAD))
    assert level == "warning"
    assert any("Signal" in r for r in reasons)


def test_non_sinus_rhythm_alone_is_warning():
    level, reasons = _patient_status(make_patient(rhythm=CardiacRhythm.ATRIAL_FIBRILLATION))
    assert level == "warning"
    assert "Atrial Fibrillation" in reasons
