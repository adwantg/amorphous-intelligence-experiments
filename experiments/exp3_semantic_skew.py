"""
Feature: Experiment 3 - Semantic Skew
Author: gadwant
Date: 2024-05-15
Description: Implements Experiment 3 to measure semantic skew in concept interpretation.
"""

import sys
import os
import json

# Add parent directory to path to import ollama_integration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.ollama_integration import AmorphousExperimentRunner

def run_experiment_3(
    models: list = None
):
    """
    Run Experiment 3: Semantic Skew using real Ollama models.
    """
    runner = AmorphousExperimentRunner(models=models)
    results = runner.experiment_3_semantic_skew()
    return results

if __name__ == "__main__":
    run_experiment_3()
