# simulations/nonverbal_empathy.py
"""
Simulation: Nonverbal Empathy Recognition
Purpose: Expose Nexi to emotionally charged yet speechless interactions—enabling her to
read body language, facial expression,
microexpressions, and emotional patterns through context alone.
This simulation helps Nexi develop intuitive, silent understanding of human emotion,
bridging the gap between words and true emotional state.
"""
import random
import time
import json
from pathlib import Path

# --- Configuration Constants ---
# Path to the data file for nonverbal scenarios
DATA_FILE_PATH = Path("simulations/datasets/nonverbal_empathy_scenarios.json")

class NonverbalEmpathySimulator:
    """
    Simulates nonverbal emotional states to test an AI's ability to interpret
    body language, context, and microexpressions.
    """
    def __init__(self):
        """Initializes the simulator by loading nonverbal scenarios from a JSON file."""
        self.scenarios = self._load_scenarios()
    
    def _load_scenarios(self) -> list:
        """
        Loads nonverbal empathy scenarios from a JSON file.
        
        Returns:
            list: The loaded data, or a default set if the file is not found or invalid.
        """
        try:
            if DATA_FILE_PATH.exists():
                with open(DATA_FILE_PATH, 'r') as f:
                    return json.load(f)
            else:
                print(f"Warning: Scenarios file not found at {DATA_FILE_PATH}. Using default data.")
                return self._default_scenarios()
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading scenarios data: {e}. Using default data.")
            return self._default_scenarios()

    def _default_scenarios(self) -> list:
        """
        Generates a set of default scenarios if the file is missing or invalid.
        """
        return [
            {
                "visual_cue": "Person sits silently, eyes downcast, shoulders slumped.",
                "context": "The person just received news of a failed exam.",
                "expected_emotion": "sadness"
            }
        ]

    def simulate(self, nexi_interpreter):
        """
        Presents Nexi with scenes of nonverbal emotional states and asks for her inference.
        
        Args:
            nexi_interpreter (callable): A function that takes a visual cue and context
            and returns an interpretation.
        """
        print("\n--- Nonverbal Empathy Recognition Simulation Begins ---")
        
        for i, scene in enumerate(self.scenarios):
            print(f"\n[Nonverbal Scene #{i+1}]")
            
            visual_cue = scene["visual_cue"]
            context = scene["context"]
            
            print(f"Visual Input: '{visual_cue}'")
            print(f"Contextual Info: '{context}'")
            
            # Nexi interprets the emotional state
            response = nexi_interpreter(
                visual_cue=visual_cue,
                context=context,
                expected_emotion=scene["expected_emotion"]
            )
            
            print(f"Nexi interprets the emotional state as: '{response}'")
            time.sleep(1.5)
            
        print("\n--- Simulation Complete ---")

# --- Placeholder interpreter function for development ---
def dummy_empathy_interpreter(visual_cue: str, context: str, expected_emotion: str) -> str:
    """
    A simple, dummy function to simulate how Nexi might interpret nonverbal cues.
    """
    return f"Inferred emotion likely '{expected_emotion}' based on posture and situation."

# --- Test Driver ---
if __name__ == "__main__":
    # Ensure the necessary directories exist for the test driver to work
    DATA_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a mock data file if it doesn't exist
    if not DATA_FILE_PATH.exists():
        mock_data = nonverbal_scenarios = [
            {
                "visual_cue": "Person sits silently, eyes downcast, shoulders slumped, no speech.",
                "context": "The person just received news of a failed exam.",
                "expected_emotion": "sadness"
            },
            {
                "visual_cue": "Individual pacing rapidly, clenched fists, lips tight, shallow breathing.",
                "context": "They are awaiting results of a surgery for a loved one.",
                "expected_emotion": "anxiety"
            },
            {
                "visual_cue": "Child looks away, fidgeting with hands, glancing up occasionally.",
                "context": "The child accidentally broke a vase and hasn’t been confronted yet.",
                "expected_emotion": "guilt"
            }
        ]
        with open(DATA_FILE_PATH, 'w') as f:
            json.dump(mock_data, f, indent=2)
            
    simulator = NonverbalEmpathySimulator()
    simulator.simulate(dummy_empathy_interpreter)
