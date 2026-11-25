# Alarm comparison: threshold rule vs logistic regression

Script: `evaluation/compare_alarm_methods.py`. Simulated data only. Not clinically validated.

## Setup
- 20,000 simulated seconds per seed, 8 patients, seeds 42, 7, and 2026.
- Labels come from the simulator's own state (real rhythm event, technical artifact, or quiet).
- Neither method sees the labels. Both see heart rate, SpO2, systolic BP, and respiratory rate only.
- Baseline: alarm when the last 5 seconds all have HR < 50, HR > 110, or SpO2 < 92.
- Model: logistic regression on 10-second window features, trained on patients 1 to 6 and tested on patients 7 and 8.

## Results (test patients, average of 3 seeds)

| Method   | Recall of real events | False alarms per patient-hour |
|----------|-----------------------|-------------------------------|
| Baseline | 34.0%                 | 4.38                          |
| Model    | 97.9%                 | 10.56                         |

The model finds more real events but raises more false alarm episodes. It does not reduce false alarms in this setup.

## Caveats
- The simulator puts a non-sinus rhythm in about 40% of seconds. Real ICUs are far quieter, so these numbers will not transfer directly.
- The baseline threshold is my choice, not a clinical standard.
- Only one train/test split and a fixed 0.5 probability cutoff were used. A threshold sweep is the next step.
- The server's built-in triage reads the simulator's hidden rhythm and artifact labels, so it cannot serve as a fair baseline.
