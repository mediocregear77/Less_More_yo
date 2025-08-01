import json
import datetime
from pathlib import Path

# --- Configuration Constants ---
# Path for saving and loading the flow state.
FLOW_STATE_PATH = Path("growth_core/flow_state.json")

# Optimal flow zone thresholds.
OPTIMAL_FLOW_LOAD_MIN = 0.3
OPTIMAL_FLOW_LOAD_MAX = 0.7
OPTIMAL_FLOW_GUIDANCE_MIN = 0.2
OPTIMAL_FLOW_GUIDANCE_MAX = 0.6

# Multipliers and weights for the guidance level calculation.
TRUST_CORRECTION_FACTOR = 0.2

class FlowState:
    """
    Manages and persists the internal metrics that define the AI's flow state.

    This class holds key cognitive and emotional metrics and provides methods
    for updating, calculating, and persisting the state.
    """
    def __init__(self):
        """
        Initializes the flow state, loading from a file if it exists,
        otherwise starting with default values.
        """
        self.cognitive_load: float = 0.0      # 0.0 (low) to 1.0 (high)
        self.emotional_valence: float = 0.0   # -1.0 (negative) to 1.0 (positive)
        self.curiosity: float = 0.5           # 0.0 (low) to 1.0 (high)
        self.guidance_level: float = 0.8      # 0.0 (autonomous) to 1.0 (guided)
        self.trust_gradient: float = 1.0      # 0.0 (untrusted) to 1.0 (fully trusted)
        self.last_update: datetime.datetime = datetime.datetime.now()

        # Attempt to load a previous state
        self._load_state()

    def _load_state(self):
        """
        Loads the flow state from the JSON file.
        """
        try:
            if FLOW_STATE_PATH.exists():
                with open(FLOW_STATE_PATH, "r") as f:
                    state = json.load(f)
                    self.cognitive_load = state.get("cognitive_load", self.cognitive_load)
                    self.emotional_valence = state.get("emotional_valence", self.emotional_valence)
                    self.curiosity = state.get("curiosity", self.curiosity)
                    self.guidance_level = state.get("guidance_level", self.guidance_level)
                    self.trust_gradient = state.get("trust_gradient", self.trust_gradient)
                    # Convert timestamp string back to datetime object
                    timestamp_str = state.get("timestamp")
                    if timestamp_str:
                        self.last_update = datetime.datetime.fromisoformat(timestamp_str)
                    print(f"Flow state loaded from '{FLOW_STATE_PATH}'.")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load flow state. Starting new session. Error: {e}")

    def update_metrics(self, load: float, valence: float, curiosity: float, trust: float):
        """
        Updates the core flow metrics and recalculates the guidance level.

        Args:
            load (float): Cognitive load, from 0.0 to 1.0.
            valence (float): Emotional valence, from -1.0 to 1.0.
            curiosity (float): Curiosity level, from 0.0 to 1.0.
            trust (float): Trust gradient, from 0.0 to 1.0.
        """
        # Ensure values are within their valid ranges
        self.cognitive_load = max(0.0, min(1.0, load))
        self.emotional_valence = max(-1.0, min(1.0, valence))
        self.curiosity = max(0.0, min(1.0, curiosity))
        self.trust_gradient = max(0.0, min(1.0, trust))
        
        # Recalculate guidance based on the new metrics
        self.guidance_level = FlowCalculator.calculate_guidance(
            self.cognitive_load, self.emotional_valence, self.curiosity, self.trust_gradient
        )
        self.last_update = datetime.datetime.now()
        self.save_state()

    def is_in_optimal_flow(self) -> bool:
        """
        Checks if the AI is in a synergistic learning zone.
        """
        is_load_optimal = OPTIMAL_FLOW_LOAD_MIN <= self.cognitive_load <= OPTIMAL_FLOW_LOAD_MAX
        is_guidance_optimal = OPTIMAL_FLOW_GUIDANCE_MIN <= self.guidance_level <= OPTIMAL_FLOW_GUIDANCE_MAX
        return is_load_optimal and is_guidance_optimal

    def get_flow_profile(self) -> dict:
        """
        Returns a dictionary summary of the current flow state.
        """
        return {
            "cognitive_load": self.cognitive_load,
            "emotional_valence": self.emotional_valence,
            "curiosity": self.curiosity,
            "guidance_level": self.guidance_level,
            "trust_gradient": self.trust_gradient,
            "optimal_flow": self.is_in_optimal_flow(),
            "timestamp": self.last_update.isoformat()
        }
    
    def save_state(self):
        """
        Saves the current flow state to a JSON file.
        """
        FLOW_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(FLOW_STATE_PATH, "w") as f:
                json.dump(self.get_flow_profile(), f, indent=2)
            print(f"Flow state saved to '{FLOW_STATE_PATH}'.")
        except IOError as e:
            print(f"Error: Failed to save flow state. {e}")

class FlowCalculator:
    """
    A stateless utility class for calculating flow metrics.
    """
    @staticmethod
    def calculate_guidance(load: float, valence: float, curiosity: float, trust: float) -> float:
        """
        Harmonizes guidance level using emotion, load, curiosity, and trust.

        Args:
            load (float): The current cognitive load.
            valence (float): The current emotional valence.
            curiosity (float): The current curiosity level.
            trust (float): The current trust gradient.

        Returns:
            float: The calculated guidance level, from 0.0 to 1.0.
        """
        # Calculate stress based on cognitive load and emotional valence
        # High load and neutral/negative valence increases stress.
        stress = load * (1 - abs(valence))
        
        # Calculate the signal for autonomy
        # High curiosity and low stress increases the desire for autonomy.
        autonomy_signal = curiosity * (1 - stress)
        
        # Trust in the user reduces the need for self-correction.
        trust_correction = 1.0 - (trust * TRUST_CORRECTION_FACTOR)
        
        # Guidance is the inverse of autonomy, with a correction for trust.
        new_guidance = 1.0 - (autonomy_signal * trust_correction)
        
        return round(max(0.0, min(1.0, new_guidance)), 2)

# --- Test Driver ---
if __name__ == "__main__":
    # Ensure the directory exists
    FLOW_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove previous log to ensure a clean start for the test
    if FLOW_STATE_PATH.exists():
        FLOW_STATE_PATH.unlink()
        print(f"Removed previous flow state file for a clean test run.")

    print("--- Test Run 1: New Flow State ---")
    flow = FlowState()
    print("Initial State:", json.dumps(flow.get_flow_profile(), indent=2))
    
    # Update metrics to a state that should be in optimal flow
    print("\n--- Updating metrics to an optimal flow state ---")
    flow.update_metrics(load=0.55, valence=0.3, curiosity=0.8, trust=0.95)
    print("Updated State:", json.dumps(flow.get_flow_profile(), indent=2))

    print("\n--- Test Run 2: Loading Saved State ---")
    # Simulate a new session by creating a new FlowState instance
    new_flow = FlowState()
    print("Loaded State:", json.dumps(new_flow.get_flow_profile(), indent=2))
    
    # Update metrics to a state that should be outside optimal flow
    print("\n--- Updating metrics to a high-stress, out-of-flow state ---")
    new_flow.update_metrics(load=0.9, valence=-0.6, curiosity=0.2, trust=0.1)
    print("Final State:", json.dumps(new_flow.get_flow_profile(), indent=2))

