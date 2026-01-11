"""
Feature: Experiment 4 - Synchronization Effects
Author: gadwant
Date: 2024-05-15
Description: Implements Experiment 4 to observe knowledge propagation and synchronization limitations using real LLMs.
"""

import sys
import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Set, Tuple

# Add parent directory to path to import ollama_integration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.ollama_integration import OllamaModel, AmorphousExperimentRunner

class TextNode:
    """Node in a network with text-based belief state using LLMs."""
    
    def __init__(self, node_id: int, model: OllamaModel, initial_belief: str):
        self.node_id = node_id
        self.model = model
        self.belief = initial_belief
        self.updates_received = 0
        
    def generate_update(self):
        """Node refines its belief internally (drift)."""
        prompt = (
            f"Current belief: {self.belief}\n\n"
            "Refine this belief by adding a new specific detail or perspective based on your own reasoning. "
            "Keep it concise (1-2 sentences)."
        )
        response = self.model.generate(prompt, temperature=0.7)
        self.belief = response.response
        
    def synchronize(self, other_belief: str):
        """Merge local belief with received belief."""
        prompt = (
            f"My belief: {self.belief}\n"
            f"Peer belief: {other_belief}\n\n"
            "Synthesize these two beliefs into a single coherent statement. "
            "Resolve conflicts if any, but try to incorporate both perspectives. "
            "Keep it concise."
        )
        response = self.model.generate(prompt, temperature=0.3)
        self.belief = response.response
        self.updates_received += 1

def run_experiment_4(
    num_nodes: int = 3,
    steps: int = 5,
    models: list = None
):
    """
    Run Experiment 4: Opportunistic Synchronization using real LLMs.
    """
    if models is None:
        models = ["mistral", "llama3", "gemma3"]
        
    print(f"Running Experiment 4: Opportunistic Synchronization (Text-based)")
    print(f"Nodes: {num_nodes}, Steps: {steps}, Models: {models}")
    
    # Initialize nodes with a starting belief
    base_belief = "Edge computing requires decentralized management."
    nodes = []
    
    # Use available models for nodes (cycling if fewer models than nodes)
    for i in range(num_nodes):
        model_name = models[i % len(models)]
        ollama_model = OllamaModel(model_name)
        # Give slight variation initially
        initial_prompt = f"Paraphrase this statement: '{base_belief}'"
        resp = ollama_model.generate(initial_prompt)
        nodes.append(TextNode(i, ollama_model, resp.response))
        
    runner = AmorphousExperimentRunner() # Helper for divergence calculation
    history = []
    
    for t in range(steps):
        print(f"\nStep {t}:")
        
        # 1. Independent Operation (Drift)
        for node in nodes:
            node.generate_update()
            
        # 2. Opportunistic Sync (Simple ring topology for simulation)
        # Node i synchronizes with Node i+1
        sync_pairs = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
        
        for i, j in sync_pairs:
            # i receives from j
            nodes[i].synchronize(nodes[j].belief)
            print(f"  Node {i} synced with Node {j}")
            
        # 3. Calculate Divergence
        current_beliefs = [n.belief for n in nodes]
        divergence = runner._text_divergence(current_beliefs)
        
        print(f"  Current Divergence: {divergence:.4f}")
        history.append({
            "step": t,
            "divergence": divergence,
            "beliefs": current_beliefs
        })
        
    results = {
        "experiment": "synchronization_text",
        "history": history,
        "final_divergence": history[-1]["divergence"]
    }
    
    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    outfile = os.path.join(results_dir, "exp4_results.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Results saved to {outfile}")
    return results

if __name__ == "__main__":
    run_experiment_4()
