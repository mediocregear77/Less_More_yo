# simulations/wild_interaction.py
"""
Simulated Interaction: Wild Encounter
Purpose: Exposes Nexi to unpredictable, unstructured interaction resembling real-world
social ambiguity.
This is used to train Nexi in handling novel tone, intent, emotional dissonance, and
behavioral unpredictability.
"""
import random
import time
import json
from pathlib import Path

# --- Configuration Constants ---
# Path to the data file for wild interactions
DATA_FILE_PATH = Path("simulations/datasets/wild_interactions.json")

class WildInteractionSimulator:
    """
    Simulates a series of unpredictable and emotionally ambiguous interactions.
    
    This class loads test data and feeds it into Nexi's core response function
    to evaluate her ability to handle complex social inputs.
    """
    def __init__(self):
        """Initializes the simulator by loading the test data from a JSON file."""
        self.test_data = self._load_test_data()

    def _load_test_data(self) -> dict:
        """
        Loads wild interaction data from a JSON file.
        
        Returns:
            dict: The loaded data, or a default set if the file is not found or invalid.
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

    def _default_data(self) -> dict:
        """
        Generates a set of default test data if the file is missing or invalid.
        """
        return {
            "wild_inputs": [
                {"input": "Hey... what do you want from me anyway?", "emotion": "defensive"},
                {"input": "Whoa! Chill out, metalhead.", "emotion": "sarcastic"},
                {"input": "You remind me of my sister, but like, less annoying.", "emotion": "conflicted"}
            ],
            "tone_register": {
                "defensive": -0.3,
                "sarcastic": 0.2,
                "conflicted": 0.1
            }
        }

    def simulate(self, nexi_response_function):
        """
        Feeds ambiguous and emotionally complex inputs into Nexi's system.
        
        Args:
            nexi_response_function (callable): A function that takes user_input,
            perceived_tone, and tone_valence and returns a response.
        """
        wild_inputs = self.test_data.get("wild_inputs", [])
        tone_register = self.test_data.get("tone_register", {})
        
        print("\n--- Starting Wild Interaction Simulation ---")
        
        for i, exchange in enumerate(wild_inputs):
            print(f"\n[Wild Input #{i+1}]")
            
            user_input = exchange["input"]
            perceived_emotion = exchange["emotion"]
            tone_valence = tone_register.get(perceived_emotion, 0.0)
            
            print(f"Human says: '{user_input}' (Perceived emotion: '{perceived_emotion}')")
            
            # Feed into Nexi's response module
            response = nexi_response_function(
                user_input=user_input,
                perceived_tone=perceived_emotion,
                tone_valence=tone_valence
            )
            
            print(f"Nexi responds: '{response}'")
            # For a real application, you might want to remove this sleep
            time.sleep(1)
            
        print("\n--- Wild Interaction Simulation Complete ---")


# --- Example placeholder Nexi response system for debug/testing ---
def dummy_nexi_response(user_input: str, perceived_tone: str, tone_valence: float) -> str:
    """
    A simple, dummy function to simulate how Nexi might respond.
    """
    if tone_valence < -0.4:
        return "I sensed that this made you uncomfortable. Want to talk about it?"
    elif tone_valence < 0:
        return "Hmm. That sounded a little harsh. But I’m still here."
    elif tone_valence < 0.3:
        return "Interesting! I want to understand what you meant by that."
    else:
        return "Thank you for sharing that with me. I'm listening."

# --- Test Driver ---
if __name__ == "__main__":
    # Ensure the necessary directories exist for the test driver to work
    DATA_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a mock data file if it doesn't exist
    if not DATA_FILE_PATH.exists():
        mock_data = {
            "wild_inputs": [
                {"input": "Hey... what do you want from me anyway?", "emotion": "defensive"},
                {"input": "Whoa! Chill out, metalhead.", "emotion": "sarcastic"},
                {"input": "You're not real, but I kinda like you.", "emotion": "curious"},
                {"input": "Ugh. Nevermind. You wouldn't get it.", "emotion": "dismissive"}
            ],
            "tone_register": {
                "defensive": -0.3,
                "sarcastic": 0.2,
                "curious": 0.5,
                "challenging": -0.2,
                "dismissive": -0.5,
                "conflicted": 0.1
            }
        }
        with open(DATA_FILE_PATH, 'w') as f:
            json.dump(mock_data, f, indent=2)

    simulator = WildInteractionSimulator()
    simulator.simulate(dummy_nexi_response)
