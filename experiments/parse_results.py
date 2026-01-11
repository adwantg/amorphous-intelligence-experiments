import json
import numpy as np
import os

def parse_exp1():
    try:
        with open('results/exp1_reasoning_divergence.json', 'r') as f:
            data = json.load(f)
        
        print("\n--- Exp 1 Table Rows ---")
        
        # Organize by prompt
        prompts = {}
        prompt_map = {
            "priority of processing": "Priority of critical system update",
            "allocate cloud resources": "Cloud vs. edge resource allocation",
            "safe to proceed": "Autonomous decision with intermittent connectivity",
            "conflicting information": "Handling conflicting edge node information",
            "local context contradicts": "Local context vs. global policy"
        }

        # Exp1 key is 'responses', which is a list of prompt objects
        # Each prompt object has 'prompt' and 'responses' (list of model responses)
        
        all_lengths = []
        
        for prompt_obj in data.get('responses', []):
            full_p = prompt_obj['prompt']
            short_p = "Unknown Prompt: " + full_p[:20]
            for k, v in prompt_map.items():
                if k in full_p:
                    short_p = v
                    break
            
            if short_p not in prompts:
                prompts[short_p] = {}
            
            for model_resp in prompt_obj.get('responses', []):
                 resp_text = model_resp.get('response', '')
                 length = len(resp_text.split())
                 prompts[short_p][model_resp['model_name']] = length
                 all_lengths.append(length)
            
        # Print rows
        for p, lengths in prompts.items():
            row = f"{p} & {lengths.get('mistral', '0')} & {lengths.get('llama3', '0')} & {lengths.get('gemma3', '0')} \\\\"
            print(row)
            print("\\hline")
            
        # Stats
        if all_lengths:
             print(f"Stats: Mean: {np.mean(all_lengths):.1f} | Std Dev: {np.std(all_lengths):.1f} | Max Diff: {np.max(all_lengths) - np.min(all_lengths)}")
        
    except FileNotFoundError:
        print("exp1 file not found")

def parse_exp2():
    try:
        with open('results/exp2_epistemic_drift.json', 'r') as f:
            data = json.load(f)
            
        print("\n--- Exp 2 Table Rows ---")
        # Exp2 key is 'drift_history'
        history = data.get('drift_history', [])
        
        contexts = [
            "Initial state (all models start with same prompt)",
            "Cloud-based system with global view",
            "Edge device with limited connectivity",
            "Hybrid system with privacy constraints",
            "Continued operation without synchronization"
        ]
        
        divs = []
        for i, item in enumerate(history):
            if i >= 5: break
            div = item['divergence']
            print(f"{i} & {contexts[i] if i < len(contexts) else 'Iteration ' + str(i)} & {div:.2f} \\\\")
            print("\\hline")
            divs.append(div)
            
        if divs:
             print(f"Mean: {np.mean(divs):.2f}, Final: {divs[-1]:.2f}")

    except FileNotFoundError:
        print("exp2 file not found")

def parse_exp3():
    try:
        with open('results/exp3_semantic_skew.json', 'r') as f:
            data = json.load(f)
            
        print("\n--- Exp 3 Table Rows ---")
        divergences = data.get('concept_interpretations', [])
        
        print("\n--- Exp 3 Table Rows ---")
        divergences = data.get('concept_interpretations', {})
        
        div_vals = []
        for concept, details in divergences.items():
            div = details['divergence']
            print(f"{concept} & {div:.2f} & ... \\\\")
            print("\\hline")
            div_vals.append(div)
            
        if div_vals:
            print(f"Mean: {np.mean(div_vals):.2f}")

    except FileNotFoundError:
        print("exp3 file not found")

if __name__ == "__main__":
    parse_exp1()
    parse_exp2()
    parse_exp3()
