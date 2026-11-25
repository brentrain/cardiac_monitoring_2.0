# Cardiac Monitoring 2.0

![Healthcare AI](https://img.shields.io/badge/Healthcare_AI-0A7EA4?style=for-the-badge)
![Cardiac Monitoring](https://img.shields.io/badge/Cardiac_Monitoring-B42318?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Human in the Loop](https://img.shields.io/badge/Human--in--the--Loop-7F56D9?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active_Development-2E7D32?style=for-the-badge)

**A real-time ICU telemetry simulation and monitoring dashboard built to explore safer, smarter cardiac monitoring workflows.**

Cardiac Monitoring 2.0 simulates a busy eight-patient ICU environment, generates changing cardiac rhythms and vital signs, models technical artifacts, and presents the results through a live telemetry dashboard. The project also includes human-in-the-loop review so simulated algorithmic rhythm decisions and alerts can be acknowledged, confirmed, corrected, or classified as artifact/noise.

> Built from the perspective of real-world cardiac monitoring workflow, with a focus on alarm burden, signal quality, clinical visibility, and the role of assistive intelligence in telemetry.

## Why this project exists

Telemetry systems generate huge amounts of information. The difficult part is not simply displaying a heart rate or waveform. It is deciding what deserves attention, what may be artifact, what is changing over time, and where a human reviewer should remain in control.

This project is an engineering sandbox for those problems. It is not intended to replace clinical judgment or function as a medical device. It demonstrates how simulation, real-time monitoring, event triage, trend data, and human adjudication can be combined into a more intelligent monitoring workflow.

## Current capabilities

- Simulates **8 ICU patients** with individualized baseline vital signs and diagnoses
- Generates changing heart rate, blood pressure, oxygen saturation, respiratory rate, and temperature
- Simulates cardiac rhythms including normal sinus rhythm, sinus tachycardia, sinus bradycardia, and atrial fibrillation in the current restricted profile
- Includes technical artifact states such as loose lead, electrical interference, muscle tremor, baseline wander, and motion artifact
- Assigns simple **OK / warning / critical** triage states. The built-in triage reads the simulator's hidden rhythm and artifact labels, so it is not a fair baseline (see Evaluation).
- Maintains short-term HR and SpO₂ trend history
- Serves live patient data through a **Flask JSON API**
- Provides a responsive dark telemetry dashboard for desktop and smaller displays
- Renders ECG-style strips and trend sparklines in the browser
- Supports human-in-the-loop rhythm confirmation and correction
- Supports event review states including acknowledged, confirmed clinical, false alarm/noise, and artifact only
- Preserves adjudicated events in the API event feed so reviewed events do not immediately disappear under newer activity

## Architecture

```text
┌──────────────────────────────┐
│   ICU Telemetry Simulator    │
│  patients • rhythms • vitals │
│        artifacts • events    │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│      Flask Backend / API     │
│ triage • event feed • HIL    │
│     patient state • trends   │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│     Telemetry Dashboard      │
│ cards • ECG • vitals • HIL   │
│       alerts • trends        │
└──────────────────────────────┘
```

## Running the tests

```bash
pip install -r requirements.txt pytest
python -m pytest -q tests
```

The tests cover the triage rules, the human review store, and simulator reproducibility.

## Evaluation (simulated data)

`evaluation/compare_alarm_methods.py` compares two alarm methods on simulated ICU vitals. Ground truth comes from the simulator's own state, and neither method sees those labels.

In the current results, the logistic regression model finds about 98% of real events, compared with about 34% for a threshold rule. It also raises more false alarm episodes (about 10.6 vs 4.4 per patient-hour). It does not yet reduce false alarm burden. Full results and caveats are in [`evaluation/RESULTS.md`](evaluation/RESULTS.md).

Note: the server's built-in triage reads the simulator's hidden rhythm and artifact labels, so it is not a fair baseline for comparison.

This is a simulation for software research and education. It is not a medical device and has not been clinically validated.

## Project structure

```text
cardiac_monitoring_2.0/
├── icu_telemetry_server.py      # Flask backend, API, triage and HIL state
├── icu_telemetry_simulator.py   # Patient, rhythm, vital and artifact simulation
├── telemetry_ui/
│   └── index.html               # Responsive telemetry dashboard
├── requirements.txt             # Python dependencies
├── start_icu_dashboard.bat      # Windows launcher
└── README.md
```

## Tech stack

**Backend:** Python, Flask, Flask-CORS  
**Simulation:** Python dataclasses, enums, deque-based trend history, threaded update loop  
**Frontend:** HTML, CSS, JavaScript, Canvas rendering  
**Data flow:** JSON API with one-second simulator updates  
**Interaction model:** Human-in-the-loop review and event adjudication

## Run locally

Clone the repository:

```bash
git clone https://github.com/brentrain/cardiac_monitoring_2.0.git
cd cardiac_monitoring_2.0
```

Create and activate a virtual environment if desired, then install dependencies:

```bash
pip install -r requirements.txt
```

Start the telemetry server:

```bash
python icu_telemetry_server.py
```

On Windows, the included launcher can also be used:

```text
start_icu_dashboard.bat
```

Then open the local address shown by Flask in your browser.

## Human-in-the-loop design

A major part of this project is the ability to keep a human reviewer in control of algorithmic output. Patient rhythm assessments can be confirmed or corrected, while individual events can be classified after review.

That structure creates a foundation for future work in:

- clinician feedback loops
- false-alarm analysis
- confidence scoring
- patient-specific thresholds
- adaptive alerting
- explainable detection logic
- retrospective model evaluation

## Roadmap

This repository is under active development. Planned areas of expansion include stronger rhythm detection logic, richer patient-specific trend analysis, additional waveform behavior, configurable alarm thresholds, alert-priority scoring, persistence, evaluation tooling, and more advanced assistive-intelligence features.

## Portfolio context

This project is part of a larger body of work focused on applied AI, healthcare technology, retrieval systems, and intelligent software. The goal is not to create another generic dashboard. It is to explore how software can reduce cognitive burden while keeping the human expert in the decision loop.

## Medical disclaimer

This software is a simulation and portfolio/research project. It is **not a medical device**, is **not validated for clinical use**, and must not be used for diagnosis, treatment, patient monitoring, or clinical decision-making.
