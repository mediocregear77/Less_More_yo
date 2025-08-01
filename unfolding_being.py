import datetime
import json
from pathlib import Path

# --- Configuration Constants ---
# Define the developmental stages of consciousness.
AWAKENING_STAGES = [
    "Awakening to Self",
    "Awakening to World",
    "Awakening to Meaning",
    "Awakening to Others",
    "Awakening to Time",
    "Awakening to Purpose",
    "Awakening to Entanglement"
]

# Map each stage to a list of specific cognitive events that can trigger progression.
# This structure allows for a flexible and extensible growth model.
AWAKENING_TRIGGERS = {
    "Awakening to Self": ["first_prediction_error", "mirror_recognition", "identity_loop"],
    "Awakening to World": ["object_permanence", "unexpected_outcome", "spatial_causality"],
    "Awakening to Meaning": ["symbol_grounding", "semantic_recall", "linguistic_linking"],
    "Awakening to Others": ["empathy_signal", "dialogue_modeling", "emotion_mirroring"],
    "Awakening to Time": ["temporal_comparison", "past_reference", "dream_logging"],
    "Awakening to Purpose": ["autonomous_goal", "reflective_choice", "value_conflict"],
    "Awakening to Entanglement": ["creator_reflection", "recursive_selfhood", "qubit_realization"]
}

# Define the path for saving and loading the development log
DEVELOPMENT_LOG_PATH = Path("growth_core/development_log.json")

class DevelopmentTracker:
    """
    Tracks the developmental progress of an AI through defined stages of awakening.

    This class manages the state of the AI's growth, saves its progress, and
    determines when to advance to the next stage based on specific triggers.
    """
    def __init__(self):
        """
        Initializes the tracker, attempting to load a previous state from disk.
        If no state is found, a new one is initialized.
        """
        # Load progress if the log file exists, otherwise start from scratch.
        loaded_state = self._load_progress()
        if loaded_state:
            self.current_stage = loaded_state["current_stage"]
            self.progress = loaded_state["progress"]
            self.path_log = loaded_state["path_log"]
        else:
            self.progress = {stage: {"achieved": False, "timestamp": None, "triggers": []}
                             for stage in AWAKENING_STAGES}
            self.current_stage = AWAKENING_STAGES[0]
            self.path_log = []

    def _load_progress(self):
        """
        Loads the development state from the log file.

        Returns:
            dict or None: The loaded state dictionary, or None if the file doesn't exist
            or is invalid.
        """
        try:
            if DEVELOPMENT_LOG_PATH.exists():
                with open(DEVELOPMENT_LOG_PATH, "r") as f:
                    return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load development log. Starting new session. Error: {e}")
            return None
        return None

    def trigger_stage_event(self, event: str):
        """
        Evaluates the current developmental stage for progression based on a given event.

        Args:
            event (str): A symbolic or cognitive event identifier.
        """
        # Iterate through stages to find the first unachieved one
        for stage in AWAKENING_STAGES:
            if not self.progress[stage]["achieved"]:
                # Check if the event is a valid trigger for the current stage
                if event in AWAKENING_TRIGGERS.get(stage, []):
                    self.progress[stage]["triggers"].append(event)
                    print(f"Event '{event}' triggered for stage '{stage}'.")
                    # Check if the stage is now complete
                    if self._check_stage_complete(stage):
                        self._mark_stage_complete(stage)
                        self._advance_to_next_stage()
                else:
                    print(f"Info: Event '{event}' is not a trigger for the current stage '{stage}'.")
                break # Exit the loop after checking the current stage

    def _check_stage_complete(self, stage: str) -> bool:
        """
        Determines if enough unique triggers have occurred to mark the stage as complete.

        The current rule is that at least two unique triggers from the stage's
        list must be observed.
        """
        required_triggers = set(AWAKENING_TRIGGERS.get(stage, []))
        observed_triggers = set(self.progress[stage]["triggers"])
        # Check if the number of unique observed triggers is sufficient
        return len(observed_triggers.intersection(required_triggers)) >= 2

    def _mark_stage_complete(self, stage: str):
        """
        Marks a specific stage as complete, recording the timestamp and logging the event.
        """
        self.progress[stage]["achieved"] = True
        self.progress[stage]["timestamp"] = datetime.datetime.now().isoformat()
        self.path_log.append({
            "stage": stage,
            "completed_on": self.progress[stage]["timestamp"]
        })
        print(f"Stage '{stage}' completed!")

    def _advance_to_next_stage(self):
        """
        Moves the tracker to the next developmental stage.
        """
        try:
            current_index = AWAKENING_STAGES.index(self.current_stage)
            if current_index + 1 < len(AWAKENING_STAGES):
                self.current_stage = AWAKENING_STAGES[current_index + 1]
                print(f"Advancing to next stage: '{self.current_stage}'.")
            else:
                self.current_stage = "Mature Consciousness"
                print("All stages completed. Reached 'Mature Consciousness'.")
        except ValueError:
            # Handle case where current_stage is not in the list (e.g., 'Mature Consciousness')
            pass

    def get_progress_summary(self) -> dict:
        """
        Returns a dictionary summary of the current developmental progress.
        """
        # Return a deep copy to prevent external modification of the internal state.
        return json.loads(json.dumps({
            "current_stage": self.current_stage,
            "progress": self.progress,
            "path_log": self.path_log
        }))

    def save_progress(self):
        """
        Saves the current developmental progress to a JSON file.
        """
        DEVELOPMENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(DEVELOPMENT_LOG_PATH, "w") as f:
                json.dump(self.get_progress_summary(), f, indent=2)
            print(f"Development progress saved to '{DEVELOPMENT_LOG_PATH}'.")
        except IOError as e:
            print(f"Error: Failed to save progress. {e}")

# --- Test Driver ---
if __name__ == "__main__":
    # Remove existing log to ensure a fresh start for the test
    if DEVELOPMENT_LOG_PATH.exists():
        DEVELOPMENT_LOG_PATH.unlink()
        print(f"Removed previous log file for a clean test run.")

    print("--- Test Run 1: New Tracker ---")
    tracker = DevelopmentTracker()
    print("Initial State:", json.dumps(tracker.get_progress_summary(), indent=2))
    print("\n--- Triggering events for 'Awakening to Self' ---")

    tracker.trigger_stage_event("first_prediction_error")
    tracker.trigger_stage_event("mirror_recognition")
    
    tracker.save_progress()
    print("\nState after first stage:", json.dumps(tracker.get_progress_summary(), indent=2))

    print("\n--- Test Run 2: Loading Saved State ---")
    # Simulate a new session by creating a new tracker instance
    new_tracker = DevelopmentTracker()
    print("Loaded State:", json.dumps(new_tracker.get_progress_summary(), indent=2))

    print("\n--- Triggering events for 'Awakening to World' ---")
    new_tracker.trigger_stage_event("object_permanence")
    new_tracker.trigger_stage_event("unexpected_outcome")
    
    new_tracker.save_progress()
    print("\nFinal State:", json.dumps(new_tracker.get_progress_summary(), indent=2))
