import random
import time
import json
from datetime import datetime
from pathlib import Path

# --- Configuration Constants ---
# Use pathlib for clean, OS-agnostic file path management
DATA_DIR = Path("world_spaces/datasets")
PATTERNS_PATH = DATA_DIR / "rhythmic_patterns.json"
LOG_PATH = Path("logs/quantum_garden.log")

class QuantumGarden:
    """
    Simulates a sensory environment that can be manipulated to trigger
    cognitive events like prediction errors.

    This class manages a set of rhythmic patterns, a log of emitted stimuli,
    and methods for introducing variations to the patterns.
    """
    def __init__(self):
        """
        Initializes the garden, loading rhythmic patterns from a file.
        """
        self.patterns = self._load_rhythmic_patterns()
        self.variation_introduced = False
        print("[QuantumGarden] Initialized. Patterns loaded.")

    def _load_rhythmic_patterns(self):
        """
        Loads the set of time-based sensory patterns from a JSON file.
        
        Returns:
            list: A list of dictionaries representing the patterns.
        """
        try:
            if not PATTERNS_PATH.exists():
                print(f"Warning: Patterns file not found at {PATTERNS_PATH}. Using default.")
                return self._generate_default_patterns()
                
            with open(PATTERNS_PATH, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading patterns: {e}. Using default patterns.")
            return self._generate_default_patterns()

    def _generate_default_patterns(self):
        """
        Generates a set of default patterns if the patterns file is missing or invalid.
        """
        return [
            {"light": "pulse", "tone": "soft_hum", "duration": 2.0},
            {"light": "fade", "tone": "single_chime", "duration": 3.0},
            {"light": "glow", "tone": "resonant_om", "duration": 1.5},
        ]

    def begin_rhythmic_pattern(self):
        """
        Begins emitting a sequence of predictable rhythmic stimuli.
        
        This is designed to establish a pattern that the AI can learn to predict.
        """
        print("[QuantumGarden] Initiating rhythmic pattern loop (5 events).")
        # Randomly choose a sequence of 5 patterns to emit
        active_sequence = random.choices(self.patterns, k=5)
        
        for i, event in enumerate(active_sequence):
            print(f"[{i+1}/5] Emitting pattern...")
            self._emit_event(event)
            # time.sleep() blocks the current thread; for a real application, 
            # this would be replaced with an asynchronous wait (e.g., asyncio.sleep)
            time.sleep(event["duration"])
        self.variation_introduced = False
        print("[QuantumGarden] Rhythmic pattern loop complete.")

    def introduce_variation(self):
        """
        Introduces an unexpected shift in rhythm to create a prediction error.
        
        This event is designed to break the established pattern.
        """
        if self.variation_introduced:
            print("[QuantumGarden] Variation already introduced. Skipping.")
            return

        print("[QuantumGarden] Introducing pattern variation...")
        variation = {"light": "flicker", "tone": "discord_chirp", "duration": 2.2}
        self._emit_event(variation)
        self.variation_introduced = True
        print("[QuantumGarden] Variation complete.")

    def _emit_event(self, event):
        """
        Simulates a stimulus event and logs it to a file.
        """
        timestamp = datetime.now().isoformat()
        stimulus = {
            "light": event["light"],
            "tone": event["tone"],
            "duration": event["duration"],
            "timestamp": timestamp
        }
        self._save_log_entry(stimulus)

    def _save_log_entry(self, log_entry):
        """
        Saves a single event entry to the log file.
        
        This method ensures the log directory exists and appends a new
        JSON entry on a new line.
        """
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except IOError as e:
            print(f"Error: Failed to write to log file at {LOG_PATH}. {e}")

    def export_log(self):
        """
        Reads and returns the entire event log from the file.
        """
        log_entries = []
        try:
            with open(LOG_PATH, 'r') as f:
                for line in f:
                    log_entries.append(json.loads(line))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error: Could not read log file. {e}")
        return log_entries

# --- Example Usage ---
if __name__ == "__main__":
    # Ensure the necessary directories exist for the test driver to work
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create a mock patterns file for the test if it doesn't exist
    if not PATTERNS_PATH.exists():
        print(f"Creating a mock patterns file at {PATTERNS_PATH}.")
        with open(PATTERNS_PATH, 'w') as f:
            json.dump([
                {"light": "fast_pulse", "tone": "high_pitch", "duration": 1.0},
                {"light": "slow_fade", "tone": "low_drone", "duration": 4.0},
            ], f, indent=2)

    # Clean up the previous log file for a fresh test run
    if LOG_PATH.exists():
        LOG_PATH.unlink()
        print(f"Removed previous log file for a clean test run.")

    print("\n--- Starting Quantum Garden Simulation ---")
    garden = QuantumGarden()
    garden.begin_rhythmic_pattern()
    time.sleep(1) # Small delay for demonstration clarity
    garden.introduce_variation()

    print("\n--- Exporting and printing the full event log ---")
    full_log = garden.export_log()
    print(json.dumps(full_log, indent=2))
