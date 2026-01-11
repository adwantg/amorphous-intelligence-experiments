"""
Feature: Experiment 2 - Epistemic Drift
Author: gadwant
Date: 2024-05-15
Description: Implements Experiment 2 to simulate and measure epistemic drift over time.
"""

import sys
import os
import json

# Add parent directory to path to import ollama_integration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.ollama_integration import AmorphousExperimentRunner

def run_experiment_2(
    num_iterations: int = 10,
    models: list = None
):
    """
    Run Experiment 2: Epistemic Drift using real Ollama models.
    """
    runner = AmorphousExperimentRunner(models=models)
    results = runner.experiment_2_epistemic_drift(num_iterations=num_iterations)
    return results

if __name__ == "__main__":
    # Reduced iterations for running as main
    run_experiment_2(num_iterations=5)
