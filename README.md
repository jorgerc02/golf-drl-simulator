# Golf DRL Simulator

A privacy-safe demonstration of a custom golf environment for deep reinforcement learning.

The project models a golf hole as a Gymnasium environment and uses Proximal Policy
Optimization (PPO) to learn shot-selection strategies. The agent chooses a club, aim
direction and power while accounting for terrain, hazards, distance and putting.

## What is included

- A custom Gymnasium golf environment.
- KML parsing and conversion from geographic coordinates to a local yard-based layout.
- Shot dispersion, terrain penalties and reward logic.
- PPO training and inference examples using Stable-Baselines3.
- One deliberately fictional course layout centred on `(0, 0)`.
- Invented club statistics for demonstrating the expected CSV structure.

## Privacy and data statement

This repository has a new, independent Git history. It does **not** contain real player
measurements, TrackMan exports, real golf-course coordinates, trained models, experiment
logs, machine names, email addresses, credentials or files from the original project's
history. The sample data is synthetic and is not suitable for golf instruction or
performance analysis.

## Project structure

```text
assets/
  sample_club_data.csv   # fictional demonstration values
  sample_par4.kml        # fictional course layout near 0°, 0°
environment.py           # Gymnasium environment
golf_hole.py             # KML parsing and coordinate conversion
golf_player.py           # club-data loader
train_model.py           # PPO training example
run_model.py             # trained-model inference example
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Train an example model with:

```bash
python train_model.py
```

Training outputs are written to ignored directories and must not be committed.

## Supplying your own data

If you adapt this code, keep private measurements and course files outside Git. Pass an
explicit path to `GolfPlayer` for club data and replace `assets/sample_par4.kml` locally.
The `.gitignore` intentionally blocks common data, model and experiment-output paths.

## Limitations

This is an academic prototype rather than a validated coaching or decision-making tool.
Its physics, reward function and synthetic inputs are simplified.
