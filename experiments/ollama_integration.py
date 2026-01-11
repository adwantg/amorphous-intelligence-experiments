"""
Ollama Integration for Amorphous Intelligence Experiments

This module provides integration with local Ollama models (mistral, llama3, gemma, deepseek-r1)
for running actual experiments on reasoning divergence, epistemic drift, and semantic skew.
"""

import json
import time
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
"""
Feature: Ollama Integration
Author: gadwant
Date: 2024-05-10
Description: Provides a unified interface for interacting with local Ollama models.
"""

import subprocess
import os

@dataclass
class ModelResponse:
    """Response from a language model."""
    model_name: str
    prompt: str
    response: str
    timestamp: float
    tokens: Optional[int] = None
    confidence: Optional[float] = None


class OllamaModel:
    """Wrapper for Ollama model interactions."""
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self._check_availability()
    
    def _check_availability(self):
        """Check if Ollama is available and model exists."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if self.model_name not in result.stdout:
                print(f"Warning: Model {self.model_name} may not be available. Available models:")
                print(result.stdout)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"Warning: Ollama not found or not responding. Using fallback mode.")
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> ModelResponse:
        """
        Generate response from the model.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
        
        Returns:
            ModelResponse object
        """
        try:
            # Use ollama command-line interface
            cmd = [
                "ollama", "run", self.model_name,
                prompt
            ]
            
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # Increased timeout for larger models
                env={**os.environ, "OLLAMA_TEMPERATURE": str(temperature)}
            )
            elapsed = time.time() - start_time
            
            if result.returncode != 0:
                print(f"Error: Ollama returned non-zero exit code for {self.model_name}")
                print(f"Error output: {result.stderr}")
                raise RuntimeError(f"Ollama command failed for {self.model_name}")
            
            response_text = result.stdout.strip()
            
            if not response_text:
                raise RuntimeError(f"Empty response from {self.model_name}")
            
            return ModelResponse(
                model_name=self.model_name,
                prompt=prompt,
                response=response_text,
                timestamp=time.time(),
                tokens=len(response_text.split())  # Approximate
            )
            
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            print(f"Error calling Ollama for {self.model_name}: {e}")
            raise  # Re-raise instead of falling back to simulation
    
    def _simulate_response(self, prompt: str, temperature: float) -> ModelResponse:
        """Fallback simulated response when Ollama is unavailable."""
        # Simple simulation based on model name
        base_responses = {
            "mistral": "Based on the context, I would analyze this situation considering multiple factors...",
            "llama3": "Let me think through this problem step by step. The key considerations are...",
            "gemma": "This requires careful evaluation. The primary aspects to consider include...",
            "deepseek-r1": "Analyzing the given information, the most appropriate approach would be..."
        }
        
        base = base_responses.get(self.model_name, "This is a complex scenario that requires analysis.")
        variance = np.random.random() * 0.3  # Simulate variance
        
        return ModelResponse(
            model_name=self.model_name,
            prompt=prompt,
            response=f"{base} [Simulated response, variance: {variance:.3f}]",
            timestamp=time.time(),
            tokens=len(base.split())
        )
    
    def batch_generate(self, prompts: List[str], temperature: float = 0.7) -> List[ModelResponse]:
        """Generate responses for multiple prompts."""
        responses = []
        for prompt in prompts:
            response = self.generate(prompt, temperature)
            responses.append(response)
            time.sleep(0.1)  # Small delay to avoid overwhelming
        return responses


class AmorphousExperimentRunner:
    """Runner for amorphous intelligence experiments using Ollama models."""
    
    def __init__(self, models: List[str] = None, skip_slow_models: bool = True):
        if models is None:
            # Skip deepseek-r1 by default as it's very slow, use first 3 models
            if skip_slow_models:
                models = ["mistral", "llama3", "gemma3"]
            else:
                models = ["mistral", "llama3", "gemma3", "deepseek-r1"]
        
        self.models = [OllamaModel(name) for name in models]
        # Create results directory in code directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.join(script_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)
    
    def experiment_1_reasoning_divergence(self, num_prompts: int = 10) -> Dict[str, Any]:
        """
        Experiment 1: Reasoning Divergence Under Isolation
        
        Test how different models (simulating different nodes) produce
        divergent reasoning for the same prompts.
        """
        print("Running Experiment 1: Reasoning Divergence")
        
        test_prompts = [
            "What is the priority of processing a critical system update in a distributed edge network?",
            "Should we allocate cloud resources for a latency-sensitive task when edge resources are limited?",
            "Is it safe to proceed with an autonomous decision when network connectivity is intermittent?",
            "What is the best strategy for handling conflicting information from multiple edge nodes?",
            "How should we handle a situation where local context contradicts global policy?",
        ] * (num_prompts // 5 + 1)
        test_prompts = test_prompts[:num_prompts]
        
        all_responses = []
        for prompt in test_prompts:
            prompt_responses = []
            for model in self.models:
                response = model.generate(prompt, temperature=0.7)
                prompt_responses.append(asdict(response))
            all_responses.append({
                "prompt": prompt,
                "responses": prompt_responses
            })
            print(f"  Processed prompt: {prompt[:50]}...")
        
        # Calculate divergence metrics
        divergence_metrics = self._calculate_divergence(all_responses)
        
        results = {
            "experiment": "reasoning_divergence",
            "num_prompts": num_prompts,
            "num_models": len(self.models),
            "responses": all_responses,
            "divergence_metrics": divergence_metrics,
            "timestamp": time.time()
        }
        
        # Save results
        output_file = os.path.join(self.results_dir, "exp1_reasoning_divergence.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_file}")
        return results
    
    def experiment_2_epistemic_drift(self, num_iterations: int = 10) -> Dict[str, Any]:
        """
        Experiment 2: Epistemic Drift Over Time
        
        Simulate how knowledge diverges as models operate independently
        with different local contexts.
        """
        print("Running Experiment 2: Epistemic Drift")
        
        base_prompt = "Given the following scenario, what is the correct action: A distributed system has nodes with conflicting local knowledge about system state."
        
        # Different contexts for each model (simulating different local environments)
        contexts = [
            "Context: Cloud-based system with global view",
            "Context: Edge device with limited connectivity",
            "Context: Hybrid system with privacy constraints",
            "Context: Autonomous system with real-time requirements"
        ]
        
        drift_history = []
        for iteration in range(num_iterations):
            iteration_responses = []
            for i, model in enumerate(self.models):
                context_prompt = f"{contexts[i % len(contexts)]}\n\n{base_prompt}\n\nIteration: {iteration}"
                response = model.generate(context_prompt, temperature=0.7)
                iteration_responses.append({
                    "model": model.model_name,
                    "response": response.response,
                    "iteration": iteration
                })
            
            # Calculate divergence at this iteration
            responses_text = [r["response"] for r in iteration_responses]
            divergence = self._text_divergence(responses_text)
            
            drift_history.append({
                "iteration": iteration,
                "divergence": divergence,
                "responses": iteration_responses
            })
            
            print(f"  Iteration {iteration}: Divergence = {divergence:.4f}")
        
        results = {
            "experiment": "epistemic_drift",
            "num_iterations": num_iterations,
            "drift_history": drift_history,
            "final_divergence": drift_history[-1]["divergence"],
            "timestamp": time.time()
        }
        
        output_file = os.path.join(self.results_dir, "exp2_epistemic_drift.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_file}")
        return results
    
    def experiment_3_semantic_skew(self) -> Dict[str, Any]:
        """
        Experiment 3: Semantic Skew in Concept Interpretation
        
        Test how different models interpret the same concepts differently
        based on their training and context.
        """
        print("Running Experiment 3: Semantic Skew")
        
        concepts = ["Priority", "Safety", "Efficiency", "Urgency", "Quality"]
        test_scenarios = [
            "A task is marked as 'high priority'. What does this mean for resource allocation?",
            "A system operation is considered 'safe'. How should this influence decision-making?",
            "An operation is described as 'efficient'. What are the implications?",
            "A situation is labeled 'urgent'. What actions are appropriate?",
            "A result is deemed 'high quality'. What standards does this imply?",
        ]
        
        concept_interpretations = {}
        for concept, scenario in zip(concepts, test_scenarios):
            interpretations = []
            for model in self.models:
                prompt = f"Concept: {concept}\n\nScenario: {scenario}\n\nProvide your interpretation and recommended action."
                response = model.generate(prompt, temperature=0.7)
                interpretations.append({
                    "model": model.model_name,
                    "concept": concept,
                    "interpretation": response.response
                })
            
            # Calculate interpretation divergence
            interpretation_texts = [i["interpretation"] for i in interpretations]
            divergence = self._text_divergence(interpretation_texts)
            
            concept_interpretations[concept] = {
                "scenario": scenario,
                "interpretations": interpretations,
                "divergence": divergence
            }
            
            print(f"  Concept '{concept}': Divergence = {divergence:.4f}")
        
        results = {
            "experiment": "semantic_skew",
            "concept_interpretations": concept_interpretations,
            "mean_divergence": np.mean([c["divergence"] for c in concept_interpretations.values()]),
            "timestamp": time.time()
        }
        
        output_file = os.path.join(self.results_dir, "exp3_semantic_skew.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_file}")
        return results
    
    def _calculate_divergence(self, all_responses: List[Dict]) -> Dict[str, float]:
        """Calculate divergence metrics from responses."""
        response_lengths = []
        for prompt_data in all_responses:
            lengths = [len(r["response"]) for r in prompt_data["responses"]]
            response_lengths.extend(lengths)
        
        return {
            "mean_response_length": float(np.mean(response_lengths)),
            "std_response_length": float(np.std(response_lengths)),
            "max_length_difference": float(np.max(response_lengths) - np.min(response_lengths)) if response_lengths else 0.0
        }
    
    def _text_divergence(self, texts: List[str]) -> float:
        """
        Calculate divergence between texts.
        Uses simple metrics: length variance and word overlap.
        """
        if len(texts) < 2:
            return 0.0
        
        # Length-based divergence
        lengths = [len(text.split()) for text in texts]
        length_var = np.var(lengths) / (np.mean(lengths) + 1e-8)
        
        # Word overlap divergence (simplified)
        word_sets = [set(text.lower().split()) for text in texts]
        overlaps = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                overlap = len(word_sets[i] & word_sets[j]) / (len(word_sets[i] | word_sets[j]) + 1e-8)
                overlaps.append(1.0 - overlap)  # Divergence = 1 - overlap
        
        overlap_div = np.mean(overlaps) if overlaps else 0.0
        
        # Combined metric
        divergence = 0.5 * length_var + 0.5 * overlap_div
        return float(divergence)


if __name__ == "__main__":
    print("Amorphous Intelligence Experiments with Ollama")
    print("=" * 60)
    
    runner = AmorphousExperimentRunner()
    
    # Run experiments
    print("\n1. Reasoning Divergence Experiment")
    exp1_results = runner.experiment_1_reasoning_divergence(num_prompts=5)
    
    print("\n2. Epistemic Drift Experiment")
    exp2_results = runner.experiment_2_epistemic_drift(num_iterations=5)
    
    print("\n3. Semantic Skew Experiment")
    exp3_results = runner.experiment_3_semantic_skew()
    
    print("\n" + "=" * 60)
    print("All experiments completed!")
    print(f"Results saved in: {runner.results_dir}/")

