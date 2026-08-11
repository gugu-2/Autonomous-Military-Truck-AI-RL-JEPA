# Developer Onboarding Guide

Welcome to the OMNIDRIVE engineering team! This document will guide you through setting up your local development environment to contribute to the codebase.

## 1. Local Environment Setup

Because OMNIDRIVE relies on highly specific GPU physics and Deep Learning architectures, you must set up your environment carefully to avoid CUDA conflicts.

### Prerequisites
* **OS:** Ubuntu 22.04 LTS (Native or via WSL2)
* **GPU:** NVIDIA RTX series (Minimum 8GB VRAM for inference, 24GB for training)
* **Drivers:** NVIDIA Driver > 535.0, CUDA Toolkit 12.4
* **Python:** 3.10+

### Setup Steps
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/gugu-2/Autonomous-Military-Truck-AI-RL-JEPA.git
   cd Autonomous-Military-Truck-AI-RL-JEPA
   ```

2. **Create a Virtual Environment:**
   Do NOT install these packages globally.
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   First, install the specific PyTorch wheel for your CUDA version.
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   pip install -e .[dev,test]
   ```
   *(Note: The `[dev]` flag installs `ruff`, `black`, and `pytest`)*

4. **Install Pre-Commit Hooks:**
   We enforce strict linting before any code can be committed.
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## 2. Codebase Structure

Before writing code, familiarize yourself with the directory layout:
* `src/jepa_brain/`: The Vision Transformer (ViT) perception and imagination world models.
* `src/rl_controller/`: The DreamerV3 reinforcement learning agent (Actor-Critic).
* `src/safety/`: Hard-coded CPU physics checks (Watchdog, Emergency Brake).
* `tests/unit/`: PyTorch mathematical validation tests.

## 3. Running the Test Suite Locally

Before opening a Pull Request, you must ensure you haven't broken the mathematical foundations of the AI.

Run the unit tests locally:
```bash
python -m pytest tests/unit/
```

If you see errors related to `Cannot import 'omegaconf'`, ensure you ran `pip install -e .[test]` successfully.

## 4. Setting up the Simulator (CARLA)

For local execution and visual debugging, you need the CARLA simulator running.
1. Download CARLA 0.9.15 from the official GitHub releases.
2. Unzip it to a directory (e.g. `~/carla-0.9.15/`).
3. Run the server:
   ```bash
   cd ~/carla-0.9.15/
   ./CarlaUE4.sh -quality-level=Epic -world-port=2000
   ```
4. In your OMNIDRIVE terminal, connect the RL agent to the simulator:
   ```bash
   python src/main.py --mode simulation
   ```
