# Space Debris Management (MARL)

This repository contains a Multi-Agent Reinforcement Learning (MARL) system designed for managing and mitigating space debris in orbital environments.

## Overview

The project provides an interactive simulation and a dashboard to analyze space debris management strategies using reinforcement learning.

### Key Components

- **`orbital_env.py`**: Contains the core simulation environment modeling space debris trajectories and orbital dynamics.
- **`train_marl.py`**: The training script that runs the multi-agent reinforcement learning models for debris mitigation.
- **`dashboard.py`**: An interactive dashboard to visualize training progress and simulation results.
- **`fetch_data.py`**: Scripts to gather and process necessary orbital data.
- **`visualize.py`**: Tools and utilities for rendering orbital scenarios and agent behaviors.

## Setup & Installation

To run this project, make sure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

There are also provided setup scripts that you can run based on your operating system:
- `setup_and_test.ps1` (for Windows PowerShell)
- `setup_and_test.sh` (for Linux/macOS)

## Usage

1. **Dashboard:** You can quickly launch the visual dashboard by running:
   ```powershell
   .\launch_dashboard.ps1
   ```
2. **Training Agents:** To train the reinforcement learning models, execute:
   ```bash
   python train_marl.py
   ```
