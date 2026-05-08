#!/usr/bin/env python3
"""
ICU Telemetry web backend: runs the simulator and serves a dashboard + JSON API.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from icu_telemetry_simulator import (
    ICUTelemetrySimulator,
    CardiacRhythm,
    TechnicalArtifact,
    rhythm_choices_for_human_ui,
    SINUS_RHYTHM_VARIANTS_ONLY,
)

UPDATE_INTERVAL_SEC = 1.0
UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telemetry_ui")

# API event feed: recent global tail plus any triaged rows still worth showing (so disposition
# does not disappear from the patient card when other beds flood the log).
API_EVENT_TAIL = 120
API_PRESERVE_ADJUDICATED_DEPTH = 450

RHYTHM_LABELS: List[str] = rhythm_choices_for_human_ui()
_VALID_RHYTHM_SET = set(RHYTHM_LABELS)

# Verdicts for events (noise reduction + review state)
EVENT_PENDING = "pending"
EVENT_ACKNOWLEDGED = "acknowledged"
EVENT_CONFIRMED_CLINICAL = "confirmed_clinical"
EVENT_FALSE_ALARM = "false_alarm_noise"
EVENT_ARTIFACT_ONLY = "artifact_only"


class HumanInLoopStore:
    """Human adjudication layered on top of the simulator (thread-safe)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._patient: Dict[int, Dict[str, Any]] = {}
        self._event: Dict[int, str] = {}

    def adjudication_for_patient(self, patient_id: int) -> Dict[str, Any]:
        with self._lock:
            raw = self._patient.get(patient_id)
            return dict(raw) if raw else {}

    def confirm_algorithm_rhythm(self, patient_id: int) -> None:
        ts = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            self._patient[patient_id] = {
                "rhythm_agrees": True,
                "human_rhythm": None,
                "updated_at": ts,
            }

    def correct_rhythm(self, patient_id: int, human_rhythm: str) -> bool:
        if human_rhythm not in _VALID_RHYTHM_SET:
            return False
        ts = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            self._patient[patient_id] = {
                "rhythm_agrees": False,
                "human_rhythm": human_rhythm,
                "updated_at": ts,
            }
        return True

    def clear_patient(self, patient_id: int) -> None:
        with self._lock:
            self._patient.pop(patient_id, None)

    def set_event_verdict(self, event_index: int, verdict: str) -> bool:
        allowed = {
            EVENT_PENDING,
            EVENT_ACKNOWLEDGED,
            EVENT_CONFIRMED_CLINICAL,
            EVENT_FALSE_ALARM,
            EVENT_ARTIFACT_ONLY,
        }
        if verdict not in allowed:
            return False
        with self._lock:
            if verdict == EVENT_PENDING:
                self._event.pop(event_index, None)
            else:
                self._event[event_index] = verdict
        return True

    def event_verdict(self, event_index: int) -> str:
        with self._lock:
            return self._event.get(event_index, EVENT_PENDING)

    def export_copy(self) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, str]]:
        with self._lock:
            return (
                {k: dict(v) for k, v in self._patient.items()},
                dict(self._event),
            )

    def clear_all(self) -> None:
        with self._lock:
            self._patient.clear()
            self._event.clear()


hil = HumanInLoopStore()


def _patient_status(patient) -> Tuple[str, List[str]]:
    """Derive a simple triage level for the dashboard."""
    reasons: List[str] = []
    level = "ok"
    v = patient.vitals
    rhythm = patient.current_rhythm
    artifact = patient.current_artifact

    if v.heart_rate == 0:
        return "critical", ["No pulse / asystole on monitor"]

    if rhythm in (CardiacRhythm.ASYSTOLE, CardiacRhythm.VENTRICULAR_FIBRILLATION):
        return "critical", [rhythm.value]

    if v.o2_saturation < 90:
        level = "critical"
        reasons.append(f"Hypoxemia SpO₂ {v.o2_saturation:.0f}%")
    elif v.o2_saturation < 92:
        level = "warning" if level == "ok" else level
        reasons.append(f"Low SpO₂ {v.o2_saturation:.0f}%")

    if v.heart_rate > 140:
        level = "warning" if level == "ok" else level
        reasons.append(f"Tachycardia {v.heart_rate:.0f} bpm")
    if v.heart_rate < 40:
        level = "warning" if level == "ok" else level
        reasons.append(f"Bradycardia {v.heart_rate:.0f} bpm")

    if v.systolic_bp < 90:
        level = "warning" if level == "ok" else level
        reasons.append(f"Hypotension SBP {v.systolic_bp:.0f}")
    if v.systolic_bp > 180:
        level = "warning" if level == "ok" else level
        reasons.append(f"Hypertension SBP {v.systolic_bp:.0f}")

    if artifact != TechnicalArtifact.NONE:
        if level == "ok":
            level = "warning"
        reasons.append(f"Signal: {artifact.value}")

    if not reasons and rhythm != CardiacRhythm.NORMAL_SINUS:
        if level == "ok":
            level = "warning"
        reasons.append(rhythm.value)

    return level, reasons


class SimulatorManager:
    def __init__(self) -> None:
        self.simulator = ICUTelemetrySimulator(num_patients=8)
        self._lock = threading.Lock()
        self.is_running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def reset_session(self) -> None:
        """New simulator instance and clear human triage state (fresh 'cache')."""
        with self._lock:
            self.simulator = ICUTelemetrySimulator(num_patients=8)
        hil.clear_all()

    def _run(self) -> None:
        while self.is_running:
            with self._lock:
                self.simulator.update_step()
            time.sleep(UPDATE_INTERVAL_SEC)

    @staticmethod
    def _patient_payload_unlocked(patient) -> Dict[str, Any]:
        status, reasons = _patient_status(patient)
        hist = list(patient.vital_history)[-36:]
        hr_trend = [round(x.heart_rate, 1) for x in hist]
        spo2_trend = [round(x.o2_saturation, 1) for x in hist]
        return {
            "id": patient.patient_id,
            "displayId": f"P{patient.patient_id:02d}",
            "name": patient.name,
            "age": patient.age,
            "diagnosis": patient.diagnosis,
            "vitals": {
                "heartRate": round(patient.vitals.heart_rate, 1),
                "systolicBP": round(patient.vitals.systolic_bp, 1),
                "diastolicBP": round(patient.vitals.diastolic_bp, 1),
                "spo2": round(patient.vitals.o2_saturation, 1),
                "respiratoryRate": round(patient.vitals.respiratory_rate, 1),
                "temperature": round(patient.vitals.temperature, 1),
            },
            "rhythm": patient.current_rhythm.value,
            "artifact": patient.current_artifact.value
            if patient.current_artifact != TechnicalArtifact.NONE
            else None,
            "status": status,
            "statusReasons": reasons,
            "hrTrend": hr_trend,
            "spo2Trend": spo2_trend,
        }

    @staticmethod
    def _build_events_for_api(
        sim: ICUTelemetrySimulator,
        event_adj: Dict[int, str],
        total_events: int,
    ) -> List[Dict[str, Any]]:
        """Recent monitor rows for all beds, plus triaged rows that left the short global tail."""
        if total_events <= 0:
            return []
        tail_start = max(0, total_events - API_EVENT_TAIL)
        preserve_start = max(0, total_events - API_PRESERVE_ADJUDICATED_DEPTH)
        indices: set[int] = set(range(tail_start, total_events))
        for idx, verdict in event_adj.items():
            if verdict == EVENT_PENDING:
                continue
            try:
                gi = int(idx)
            except (TypeError, ValueError):
                continue
            if preserve_start <= gi < total_events:
                indices.add(gi)
        log = sim.event_log
        out: List[Dict[str, Any]] = []
        for global_idx in sorted(indices, reverse=True):
            e = log[global_idx]
            verdict = event_adj.get(global_idx, EVENT_PENDING)
            out.append(
                {
                    "eventIndex": global_idx,
                    "timestamp": e.timestamp,
                    "patientId": e.patient_id,
                    "type": e.event_type,
                    "description": e.description,
                    "humanVerdict": verdict,
                    "vitals": {
                        "heartRate": round(e.vital_signs.heart_rate, 1),
                        "systolicBP": round(e.vital_signs.systolic_bp, 1),
                        "diastolicBP": round(e.vital_signs.diastolic_bp, 1),
                        "spo2": round(e.vital_signs.o2_saturation, 1),
                    },
                }
            )
        return out

    def snapshot(self) -> Dict[str, Any]:
        patient_adj, event_adj = hil.export_copy()

        with self._lock:
            sim = self.simulator
            time_step = sim.time_step
            patients_out = []
            critical_n = warning_n = 0
            num_patients = len(sim.patients)
            for p in sim.patients:
                payload = self._patient_payload_unlocked(p)
                if payload["status"] == "critical":
                    critical_n += 1
                elif payload["status"] == "warning":
                    warning_n += 1

                pid = payload["id"]
                adj = patient_adj.get(pid, {})
                agrees = adj.get("rhythm_agrees")
                human_r = adj.get("human_rhythm")
                payload["displayRhythm"] = human_r if human_r else payload["rhythm"]
                payload["humanReview"] = {
                    "hasEntry": pid in patient_adj,
                    "rhythmAgreesWithMonitor": agrees,
                    "humanRecordedRhythm": human_r,
                    "updatedAt": adj.get("updated_at"),
                    "needsReview": pid not in patient_adj,
                }

                patients_out.append(payload)

            total_events = len(sim.event_log)
            events_out = self._build_events_for_api(sim, event_adj, total_events)

        beds_reviewed = len([k for k in patient_adj if isinstance(k, int) and 1 <= k <= num_patients])
        beds_pending_review = max(0, num_patients - beds_reviewed)

        ev_confirmed = ev_false = ev_art = ev_ack = ev_pend = 0
        for i in range(total_events):
            v = event_adj.get(i, EVENT_PENDING)
            if v == EVENT_CONFIRMED_CLINICAL:
                ev_confirmed += 1
            elif v == EVENT_FALSE_ALARM:
                ev_false += 1
            elif v == EVENT_ARTIFACT_ONLY:
                ev_art += 1
            elif v == EVENT_ACKNOWLEDGED:
                ev_ack += 1
            else:
                ev_pend += 1

        beds_confirmed = sum(
            1 for pid, adj in patient_adj.items() if adj.get("rhythm_agrees") is True
        )
        beds_corrected = sum(1 for adj in patient_adj.values() if adj.get("human_rhythm"))

        return {
            "timeStep": time_step,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "patients": patients_out,
            "events": events_out,
            "rhythmOptions": RHYTHM_LABELS,
            "stats": {
                "totalEvents": total_events,
                "patientsCritical": critical_n,
                "patientsWarning": warning_n,
                "simProfile": {"sinusRhythmVariantsOnly": SINUS_RHYTHM_VARIANTS_ONLY},
                "hil": {
                    "bedsConfirmedMonitor": beds_confirmed,
                    "bedsHumanCorrected": beds_corrected,
                    "bedsPendingRhythmReview": beds_pending_review,
                    "eventsConfirmedClinical": ev_confirmed,
                    "eventsAcknowledged": ev_ack,
                    "eventsSuppressedNoise": ev_false + ev_art,
                    "eventsAwaitingReview": ev_pend,
                },
            },
        }


def build_human_in_loop_report() -> Tuple[str, str]:
    """Clinician-facing report: uses human triage; suppressed noise listed separately."""
    patient_adj, event_adj = hil.export_copy()
    with manager._lock:
        sim = manager.simulator
        patients = list(sim.patients)
        events = list(sim.event_log)

    lines: List[str] = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append("ICU TELEMETRY — CLINICIAN REPORT (human-in-the-loop)")
    lines.append(f"Generated on demand: {ts}")
    lines.append("")
    lines.append(
        "Buttons map to: REAL (confirmed clinical), NOT REAL (false alarm), ARTIFACT, "
        "or REVIEWED ONLY (seen but real vs not deferred). Suppressed = noise appendix."
    )
    lines.append("")
    lines.append("=== RHYTHM ADJUDICATION BY BED ===")

    for p in patients:
        pid = p.patient_id
        adj = patient_adj.get(pid, {})
        lines.append("")
        lines.append(f"Bed {pid:02d} — {p.name}, age {p.age}; {p.diagnosis}")
        lines.append(f"  Monitor rhythm (classifier): {p.current_rhythm.value}")
        if not adj:
            lines.append("  Human rhythm review: NOT RECORDED")
        elif adj.get("rhythm_agrees") is True:
            lines.append(
                "  Human: CONFIRMS monitor rhythm is clinically acceptable for this epoch."
            )
        elif adj.get("human_rhythm"):
            lines.append(f"  Human RECORDED true rhythm: {adj['human_rhythm']}")

    confirmed: List[str] = []
    acknowledged: List[str] = []
    suppressed: List[str] = []
    pending: List[str] = []

    for idx, e in enumerate(events):
        verdict = event_adj.get(idx, EVENT_PENDING)
        base = (
            f"[{e.timestamp}] Bed {e.patient_id:02d} | {e.event_type} | {e.description}"
        )
        if verdict == EVENT_FALSE_ALARM:
            suppressed.append(f"{base}\n    (human: FALSE ALARM / not clinically relevant)")
        elif verdict == EVENT_ARTIFACT_ONLY:
            suppressed.append(f"{base}\n    (human: ARTIFACT ONLY — treat as noise)")
        elif verdict == EVENT_CONFIRMED_CLINICAL:
            confirmed.append(
                f"{base}\n    HR {e.vital_signs.heart_rate:.0f}  SpO2 {e.vital_signs.o2_saturation:.1f}%"
            )
        elif verdict == EVENT_ACKNOWLEDGED:
            acknowledged.append(
                f"{base}\n    HR {e.vital_signs.heart_rate:.0f}  SpO2 {e.vital_signs.o2_saturation:.1f}%"
                "\n    (human: ACKNOWLEDGED — reviewed/seen)"
            )
        else:
            pending.append(base)

    lines.append("")
    lines.append("=== REAL ALARMS (you judged YES — clinically real) ===")
    if not confirmed:
        lines.append("(none yet — use “Yes — real alarm” on the dashboard)")
    else:
        lines.extend(confirmed)

    lines.append("")
    lines.append("=== REVIEWED ONLY (saw the alert; real vs not not decided) ===")
    if not acknowledged:
        lines.append("(none)")
    else:
        lines.extend(acknowledged)

    lines.append("")
    lines.append("=== EVENTS AWAITING HUMAN TRIAGE ===")
    if not pending:
        lines.append("(none)")
    else:
        for row in pending[-80:]:
            lines.append(f"  • {row}")
        if len(pending) > 80:
            lines.append(f"  … {len(pending) - 80} additional pending")

    lines.append("")
    lines.append("=== APPENDIX — SUPPRESSED AS NOISE ===")
    if not suppressed:
        lines.append("(none)")
    else:
        lines.extend(suppressed)

    lines.append("")
    lines.append(
        f"Summary counts — confirmed: {len(confirmed)} | acknowledged: {len(acknowledged)} | "
        f"suppressed: {len(suppressed)} | pending triage: {len(pending)}"
    )

    text_body = "\n".join(lines)
    fname = f"icu_telemetry_clinician_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text_body)
    return fname, text_body


manager = SimulatorManager()

app = Flask(__name__, static_folder=None)
CORS(app)


def _no_store(resp):
    """Avoid stale dashboard state when the browser caches JSON snapshot responses."""
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.after_request
def _apply_no_store_for_api(response):
    if request.path.startswith("/api/"):
        return _no_store(response)
    return response


@app.route("/api/snapshot", methods=["GET"])
def api_snapshot():
    return jsonify(manager.snapshot())


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify(
        {
            "status": "healthy",
            "simulatorRunning": manager.is_running,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )


@app.route("/api/adjudicate/patient", methods=["POST"])
def api_adjudicate_patient():
    data = request.get_json(silent=True) or {}
    try:
        patient_id = int(data["patientId"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"ok": False, "error": "patientId (int) is required"}), 400
    action = data.get("action")
    if action == "confirm_monitor":
        hil.confirm_algorithm_rhythm(patient_id)
    elif action == "correct_rhythm":
        label = data.get("humanRhythm")
        if not label or not hil.correct_rhythm(patient_id, str(label)):
            return jsonify({"ok": False, "error": "humanRhythm must match a known rhythm label"}), 400
    elif action == "clear":
        hil.clear_patient(patient_id)
    else:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "action must be confirm_monitor, correct_rhythm, or clear",
                }
            ),
            400,
        )
    return jsonify({"ok": True, "snapshotHint": "refetch /api/snapshot"})


@app.route("/api/adjudicate/event", methods=["POST"])
def api_adjudicate_event():
    data = request.get_json(silent=True) or {}
    try:
        event_index = int(data["eventIndex"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"ok": False, "error": "eventIndex (int) is required"}), 400
    verdict = data.get("verdict")
    if verdict not in (
        EVENT_PENDING,
        EVENT_ACKNOWLEDGED,
        EVENT_CONFIRMED_CLINICAL,
        EVENT_FALSE_ALARM,
        EVENT_ARTIFACT_ONLY,
    ):
        return jsonify({"ok": False, "error": "invalid verdict"}), 400
    hil.set_event_verdict(event_index, verdict)
    return jsonify({"ok": True})


@app.route("/api/session/reset", methods=["POST"])
def api_session_reset():
    """Clear human adjudication and start a new simulator (fresh event log)."""
    manager.reset_session()
    return jsonify(
        {
            "ok": True,
            "message": "Session cleared: new simulator run, triage cache reset.",
        }
    )


@app.route("/api/report/generate", methods=["POST"])
def api_report_generate():
    filename, body = build_human_in_loop_report()
    return jsonify(
        {
            "ok": True,
            "filename": filename,
            "characters": len(body),
            "preview": body if len(body) < 12000 else body[:12000] + "\n…(truncated in JSON preview; see file)…",
        }
    )


@app.route("/")
def serve_index():
    return send_from_directory(UI_DIR, "index.html")


if __name__ == "__main__":
    print("ICU Telemetry dashboard backend")
    print(f"  Open http://127.0.0.1:5050  (UI + API)")
    manager.start()
    app.run(
        host="127.0.0.1",
        port=5050,
        debug=False,
        use_reloader=False,
        threaded=True,
    )
