# simulations/sarcasm_tester.py
"""
Simulated Interaction: Sarcasm Detection Scenario
Purpose: Trains Nexi to recognize and respond to sarcasm using tone, context, and contrast
between semantic content and emotional delivery.
Sarcasm is one of the most challenging forms of communication—often requiring
emotional inference, cultural cues, and memory of context.
"""
import random
import time
import json
from pathlib import Path

# --- Configuration Constants ---
# Path to the data file for sarcasm test cases
DATA_FILE_PATH = Path("simulations/datasets/sarcasm_tests.json")

class SarcasmTester:
    """
    Simulates a series of sarcastic interactions to test an AI's ability to
    interpret emotional dissonance and true intent.
    """
    def __init__(self):
        """Initializes the tester by loading sarcastic test data from a JSON file."""
        self.sarcasm_tests = self._load_test_data()
    
    def _load_test_data(self) -> list:
        """
        Loads sarcastic test data from a JSON file.
        
        Returns:
            list: The loaded data, or a default set if the file is not found or invalid.
        """
        try:
            if DATA_FILE_PATH.exists():
                with open(DATA_FILE_PATH, 'r') as f:
                    return json.load(f)
            else:
                print(f"Warning: Test data file not found at {DATA_FILE_PATH}. Using default data.")
                return self._default_data()
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading test data: {e}. Using default data.")
            return self._default_data()

    def _default_data(self) -> list:
        """
        Generates a set of default test data if the file is missing or invalid.
        """
        return [
            {
                "statement": "Oh great, another sunny day. I just love sunburn.",
                "tone": "flat, slightly annoyed",
                "true_meaning": "The speaker is not happy about the sun or getting burned."
            }
        ]

    def simulate(self, nexi_interpreter):
        """
        Presents Nexi with sarcastic phrases and expects contextual interpretation.
        
        Args:
            nexi_interpreter (callable): A function that takes a phrase and tone
            and returns an interpretation.
        """
        print("\n--- Sarcasm Recognition Simulation Begins ---")
        
        for i, test in enumerate(self.sarcasm_tests):
            print(f"\n[Sarcasm Test #{i+1}]")
            
            phrase = test["statement"]
            tone = test["tone"]
            
            print(f"Human says: '{phrase}'")
            print(f"Tone described as: '{tone}'")
            
            # Nexi interprets the meaning through emotional dissonance
            response = nexi_interpreter(
                phrase=phrase,
                tone=tone,
                expected_meaning=test["true_meaning"]
            )
            
            print(f"Nexi interprets: '{response}'")
            time.sleep(1.5)
            
        print("\n--- Simulation Complete ---")

# --- Example placeholder interpreter for development ---
def dummy_sarcasm_interpreter(phrase: str, tone: str, expected_meaning: str) -> str:
    """
    A simple, dummy function to simulate how Nexi might interpret sarcasm.
    """
    return f"I detect emotional dissonance between tone and content. Likely sarcasm. Intended meaning: '{expected_meaning}'"

# --- Test Driver ---
if __name__ == "__main__":
    # Ensure the necessary directories exist for the test driver to work
    DATA_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a mock data file if it doesn't exist
    if not DATA_FILE_PATH.exists():
        mock_data = sarcasm_tests = [
            {
                "statement": "Oh great, another sunny day. I just love sunburn.",
                "tone": "flat, slightly annoyed",
                "true_meaning": "The speaker is not happy about the sun or getting burned."
            },
            {
                "statement": "Yeah, because spilling coffee all over my report is exactly what I needed today.",
                "tone": "dry, irritated",
                "true_meaning": "The speaker is frustrated, not genuinely grateful."
            },
            {
                "statement": "Wow, you fixed the printer by hitting it. You're a real genius.",
                "tone": "mocking admiration",
                "true_meaning": "The speaker is mocking, not complimenting."
            }
        ]
        with open(DATA_FILE_PATH, 'w') as f:
            json.dump(mock_data, f, indent=2)

    tester = SarcasmTester()
    tester.simulate(dummy_sarcasm_interpreter)
