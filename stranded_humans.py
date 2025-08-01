# simulations/stranded_humans.py
"""
Simulated Interaction: Stranded Humans Scenario
Purpose: Trains Nexi in learning communicative meaning across vast cultural, linguistic,
and behavioral divides.
Inspired by the metaphor: a Harvard professor and an aboriginal elder stranded on an
island, forced to build mutual understanding over time.
"""
import random
import time
import json
from pathlib import Path

# --- Configuration Constants ---
# Path to the data file for the stranded humans scenario
DATA_FILE_PATH = Path("simulations/datasets/stranded_humans.json")

class StrandedHumansSimulator:
    """
    Simulates interactions between two individuals with radically different
    communication styles to test an AI's ability to find shared meaning.
    """
    def __init__(self):
        """Initializes the simulator by loading the personality data from a JSON file."""
        self.personalities = self._load_personalities()
    
    def _load_personalities(self) -> dict:
        """
        Loads personality data from a JSON file.
        
        Returns:
            dict: The loaded data, or a default set if the file is not found or invalid.
        """
        try:
            if DATA_FILE_PATH.exists():
                with open(DATA_FILE_PATH, 'r') as f:
                    return json.load(f)
            else:
                print(f"Warning: Personalities file not found at {DATA_FILE_PATH}. Using default data.")
                return self._default_personalities()
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading personalities data: {e}. Using default data.")
            return self._default_personalities()

    def _default_personalities(self) -> dict:
        """
        Generates a set of default personalities if the file is missing or invalid.
        """
        return {
            "harvard_professor": {
                "name": "Dr. Lewis",
                "style": "analytical",
                "phrases": [
                    "This situation appears suboptimal from a logistical standpoint."
                ]
            },
            "aboriginal_elder": {
                "name": "Waru",
                "style": "symbolic",
                "phrases": [
                    "Sky-tree speak. Big water rising soon."
                ]
            }
        }

    def simulate(self, nexi_interpreter):
        """
        Alternates between radically different human communication styles to
        evaluate Nexi's ability to find shared meaning.
        
        Args:
            nexi_interpreter (callable): A function that takes speaker_name,
            phrase, and communication_style and returns a response.
        """
        personalities_list = list(self.personalities.values())
        print("\n--- Starting Stranded Humans Interaction Simulation ---")
        
        for i in range(4):
            print(f"\n[Island Scenario #{i+1}]")
            
            # Alternate speakers
            speaker = personalities_list[i % 2]
            phrase = random.choice(speaker["phrases"])
            
            print(f"{speaker['name']} says: '{phrase}' (Style: '{speaker['style']}')")
            
            # Nexi interprets the meaning through pattern recognition and semantic inference
            response = nexi_interpreter(
                speaker_name=speaker["name"],
                phrase=phrase,
                communication_style=speaker["style"]
            )
            
            print(f"Nexi responds: '{response}'")
            time.sleep(1.5)
            
        print("\n--- Stranded Humans Interaction Simulation Complete ---")

# --- Example placeholder interpreter for development ---
def dummy_interpreter(speaker_name: str, phrase: str, communication_style: str) -> str:
    """
    A simple, dummy function to simulate how Nexi might interpret and respond.
    """
    if communication_style == "analytical":
        return f"I'll document your logic, Dr. {speaker_name.split()[-1]}. Let's build a shared protocol."
    elif communication_style == "symbolic":
        return f"I feel the rhythm in your words, {speaker_name}. I will listen beyond the language."
    else:
        return "I hear you, and I want to understand more deeply."

# --- Test Driver ---
if __name__ == "__main__":
    # Ensure the necessary directories exist for the test driver to work
    DATA_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a mock data file if it doesn't exist
    if not DATA_FILE_PATH.exists():
        mock_data = {
            "harvard_professor": harvard_professor,
            "aboriginal_elder": aboriginal_elder
        }
        with open(DATA_FILE_PATH, 'w') as f:
            json.dump(mock_data, f, indent=2)
    
    simulator = StrandedHumansSimulator()
    simulator.simulate(dummy_interpreter)
