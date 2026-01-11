# Amorphous Intelligence Experiments

**Author**: gadwant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This directory contains code for illustrative experiments demonstrating key phenomena of amorphous intelligence.

## Structure

```
code/
├── experiments/
│   ├── exp1_reasoning_divergence.py    # Experiment 1: Reasoning divergence under isolation
│   ├── exp2_epistemic_drift.py         # Experiment 2: Epistemic drift over time
│   ├── exp3_semantic_skew.py           # Experiment 3: Semantic skew in concept interpretation
│   └── exp4_synchronization.py         # Experiment 4: Opportunistic synchronization effects
├── utils/
│   └── __init__.py                      # Utility functions
└── README.md                            # This file
```

## Requirements

- Python 3.8+
- numpy
- (Optional) Actual LLM APIs for real model deployment

## Usage

Each experiment can be run independently:

```bash
# Experiment 1: Reasoning Divergence
python experiments/exp1_reasoning_divergence.py

# Experiment 2: Epistemic Drift
python experiments/exp2_epistemic_drift.py

# Experiment 3: Semantic Skew
python experiments/exp3_semantic_skew.py

# Experiment 4: Synchronization
python experiments/exp4_synchronization.py
```

## Note
These experiments use **real local LLMs** via Ollama. 
Requirements:
1. **Ollama** installed and running.
2. Models pulled: `mistral`, `llama3`, `gemma3`.

The scripts will automatically connect to your local Ollama instance (default: `http://localhost:11434`).
Execution times may vary depending on your hardware (GPU vs CPU).

## Results

Each experiment generates a JSON results file in `experiments/results/`:
- `exp1_reasoning_divergence.json`
- `exp2_epistemic_drift.json`
- `exp3_semantic_skew.json`
- `exp4_results.json`

These results can be used for analysis and visualization to support the paper's claims.

