# SABRE: Splitting Approximated Bounds for Relational Verification

## Installation Guide

### 1. Clone the repository
```
git clone https://github.com/fukky5341/sabre.git
cd sabre
```

### 2. Install Gurobi (solver)
Please install Gurobi from the official website: [gurobi installation](https://www.gurobi.com/)

Ensure that your Gurobi license is properly installed and gurobipy works in Python.

### 3. Install uv (python environment manager)
Please install by following guide: [uv installation](https://github.com/astral-sh/uv?tab=readme-ov-file#installation)


### 4. Setup python version
The project requires Python 3.12. Please install and pin the version using uv:
```
uv python install 3.12
cd [repository folder]
uv python pin 3.12
```

### 5. Create uv environment and install dependencies
```
uv sync
```
This command:
- creates a virtual environment (.venv)
- installs all dependencies from `pyproject.toml`
- ensure the environment uses Python 3.12


## Running Experiments
Tor run the main experiment:
```
uv run run_experiment_ds_ns.py
```


## Project Structure
```
sabre/
 ├─ run_experiment_ds_ns.py    # Entry point for experiments
 ├─ relational_bounds/    # Relational bound propagation modules
 ├─ relu/    # Handle ReLU transformation in relational bound propagation
 ├─ relational_split/    # Branch-and-bound with relational splitting
 ├─ individual_split/    # Branch-and-bound with individual splitting
 ├─ relational_property/    # LP formulation for relational properties
 ├─ dual/    # Dual formulation for neuron selection
 ├─ (common, data, network_converters, ...)/  # Common utilities, datasets, and network converters
 ├─ pyproject.toml    # Project dependencies
 └─ README.md
```