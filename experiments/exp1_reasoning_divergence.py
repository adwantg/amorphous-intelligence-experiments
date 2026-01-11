"""
Experiment 1: Reasoning Divergence Under Isolation

This experiment demonstrates that reasoning diverges when nodes operate 
independently without synchronization.

Objective: Show that even identical models can produce divergent reasoning
when operating independently, illustrating the fundamental challenge of 
maintaining consistency in amorphous systems.
"""
"""
Feature: Experiment 1 - Reasoning Divergence
Author: gadwant
Date: 2024-05-15
Description: Implements Experiment 1 to demonstrate reasoning divergence across independent nodes.
"""

"""
Feature: Experiment 1 - Reasoning Divergence
Author: gadwant
Date: 2024-05-15
Description: Implements Experiment 1 to demonstrate reasoning divergence across independent nodes.
"""

import sys
import os
import json

# Add parent directory to path to import ollama_integration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.ollama_integration import AmorphousExperimentRunner

def run_experiment_1(
    num_prompts: int = 10,
    models: list = None
):
    """
    Run Experiment 1: Reasoning Divergence using real Ollama models.
    """
    runner = AmorphousExperimentRunner(models=models)
    results = runner.experiment_1_reasoning_divergence(num_prompts=num_prompts)
    return results

if __name__ == "__main__":
    run_experiment_1()

