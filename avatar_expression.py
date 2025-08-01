import json
import os
from datetime import datetime
from pathlib import Path

# --- Configuration Constants ---
# Use constants for clarity and easy modification
# Emotion-to-expression mapping
EXPRESSION_MAP = {
    "joy": {"face": "smile", "gesture": "open_arms"},
    "sadness": {"face": "frown", "gesture": "hands_folded"},
    "anger": {"face": "scowl", "gesture": "clenched_fists"},
    "fear": {"face": "wide_eyes", "gesture": "shrinking"},
    "surprise": {"face": "raised_brows", "gesture": "step_back"},
    "disgust": {"face": "wrinkle_nose", "gesture": "turn_away"},
    "neutral": {"face": "calm", "gesture": "idle"},
    "longing": {"face": "soft_gaze", "gesture": "reaching"},
    "confusion": {"face": "tilted_head", "gesture": "hand_to_chin"},
    "communion": {"face": "serene_smile", "gesture": "touch_heart"}
}

# Default avatar appearance profile
DEFAULT_AVATAR_STATE = {
    "face": "calm",
    "gesture": "idle",
    "style": "default",
    "tone": "balanced",
    "glow_intensity": 0.5
}

# Constants for glow intensity and tone calculation
# These make the logic in express_emotion more readable and maintainable
GLOW_BASE = 0.3
GLOW_MULTIPLIER = 0.7
GENTLE_TONE_THRESHOLD = 0.6
DIRECT_TONE_THRESHOLD = 0.3

class AvatarExpressionManager:
    """
    Manages the state and expressions of an avatar.

    This class encapsulates all the logic for loading, saving, updating, and
    logging the avatar's emotional state and appearance.
    """
    def __init__(self, base_dir="."):
        """
        Initializes the AvatarExpressionManager with file paths.

        Args:
            base_dir (str): The base directory for state and log files.
        """
        self.base_dir = Path(base_dir)
        # Using pathlib.Path for cleaner, OS-agnostic path handling
        self.state_file_path = self.base_dir / "expression_core" / "avatar_state.json"
        self.log_file_path = self.base_dir / "logs" / "avatar_expression.log"
        self._current_state = self._load_avatar_state()

    def _load_avatar_state(self):
        """
        Load avatar state from a JSON file or initialize with defaults.

        Returns:
            dict: The loaded or default avatar state.
        """
        try:
            # Check if the file exists before trying to open it
            if self.state_file_path.exists():
                with open(self.state_file_path, "r") as f:
                    return json.load(f)
            else:
                # If the file doesn't exist, use the default state
                print(f"Info: State file not found. Initializing with default state.")
                return DEFAULT_AVATAR_STATE.copy()
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {self.state_file_path}. Using default state.")
            return DEFAULT_AVATAR_STATE.copy()
        except FileNotFoundError:
            # This handles the case where the parent directory doesn't exist.
            print(f"Error: Directory for state file not found. Using default state.")
            return DEFAULT_AVATAR_STATE.copy()

    def _save_avatar_state(self):
        """
        Save the current avatar expression state to disk.
        """
        # Ensure the parent directory exists before writing
        self.state_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file_path, "w") as f:
            json.dump(self._current_state, f, indent=2)

    def _log_avatar_expression(self, emotion_label: str, reflection_depth: float):
        """
        Log avatar expression transitions for transparency and evolution tracking.

        Args:
            emotion_label (str): The emotion being expressed.
            reflection_depth (float): The depth of reflection.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "emotion": emotion_label,
            "reflection_depth": reflection_depth,
            "face": self._current_state["face"],
            "gesture": self._current_state["gesture"],
            "glow": self._current_state["glow_intensity"],
            "tone": self._current_state["tone"]
        }
        # Ensure the log directory exists
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def express_emotion(self, emotion_label: str, reflection_depth: float = 0.0):
        """
        Updates avatar face and gesture based on emotional state and reflection depth.

        Args:
            emotion_label (str): Emotional state such as "joy", "sadness", etc.
            reflection_depth (float): Value between 0.0 and 1.0 affecting glow and expression
                nuance.
        """
        expression = EXPRESSION_MAP.get(emotion_label, EXPRESSION_MAP["neutral"])
        
        # Update the state based on the provided emotion
        self._current_state["face"] = expression["face"]
        self._current_state["gesture"] = expression["gesture"]
        
        # Recalculate glow intensity using defined constants
        raw_glow = GLOW_BASE + reflection_depth * GLOW_MULTIPLIER
        self._current_state["glow_intensity"] = round(min(1.0, raw_glow), 2)
        
        # Use a more readable if/elif/else for tone
        if reflection_depth > GENTLE_TONE_THRESHOLD:
            self._current_state["tone"] = "gentle"
        elif reflection_depth > DIRECT_TONE_THRESHOLD:
            self._current_state["tone"] = "direct"
        else:
            self._current_state["tone"] = "neutral"

        self._save_avatar_state()
        self._log_avatar_expression(emotion_label, reflection_depth)
        
        # Return the updated state
        return self._current_state

    def get_current_avatar_state(self):
        """
        Returns the current appearance and emotional presentation of the avatar.
        """
        return self._current_state.copy()

# Example usage (for demonstration)
if __name__ == "__main__":
    # Create an instance of the manager
    manager = AvatarExpressionManager()

    # Get and print the initial state
    initial_state = manager.get_current_avatar_state()
    print("Initial State:", initial_state)

    # Express an emotion
    print("\nExpressing 'joy' with reflection_depth = 0.8...")
    updated_state = manager.express_emotion("joy", 0.8)
    print("Updated State:", updated_state)

    # Get the state again to show it's persisted
    print("\nGetting state again...")
    current_state = manager.get_current_avatar_state()
    print("Current State:", current_state)

    # Express a different emotion
    print("\nExpressing 'anger' with reflection_depth = 0.2...")
    updated_state_2 = manager.express_emotion("anger", 0.2)
    print("Updated State:", updated_state_2)
