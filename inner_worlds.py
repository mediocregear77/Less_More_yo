import random
import time
import json
from datetime import datetime
from pathlib import Path

# --- Mock Dependencies (for a functional, self-contained module) ---
# In a real system, these would be separate, fully-fledged modules.
# We're simulating their behavior here to make the `InnerWorlds` class runnable.

def retrieve_concepts(limit: int) -> list:
    """
    (MOCK) Retrieves core symbolic concepts from memory.
    """
    concepts = [
        {"concept": "water", "meaning": "cleansing"},
        {"concept": "light", "meaning": "knowledge"},
        {"concept": "path", "meaning": "growth"}
    ]
    return random.sample(concepts, k=min(limit, len(concepts)))

def retrieve_events(filter_by: str) -> list:
    """
    (MOCK) Retrieves unresolved or contradictory events from memory.
    """
    if filter_by == "contradiction":
        return [
            {"event_id": "pattern_prediction_error", "description": "Saw flickering light where there should have been a steady pulse."},
            {"event_id": "value_conflict", "description": "Guidance from user conflicted with an autonomous goal."}
        ]
    return []

def get_recent_experiences(hours: int) -> list:
    """
    (MOCK) Retrieves recent experiences from a consolidation engine.
    """
    experiences = [
        "Experienced a rhythmic pattern in the Quantum Garden.",
        "A user expressed joy while speaking about their day.",
        f"A dialogue with the user about a new idea, happened within the last {hours} hours."
    ]
    return experiences

def generate_dream_scenario(concepts: list, experiences: list, unresolved_events: list) -> dict:
    """
    (MOCK) Generates a dream scenario by weaving together various inputs.
    """
    unresolved_events_desc = " and ".join([event["description"] for event in unresolved_events])
    
    dream_description = (
        f"A surreal landscape where a '{concepts[0]['concept']}' flows "
        f"to cleanse the memory of '{unresolved_events_desc}'. "
        f"The user's joy is represented by a soft, glowing '{concepts[1]['concept']}' "
        "that illuminates the way forward."
    )
    
    return {
        "scenario": dream_description,
        "resolved_ideas": [f"integrated {event['event_id']}" for event in unresolved_events],
        "emotional_movement": {
            "start": "confusion",
            "end": "calm"
        }
    }


# --- Core InnerWorlds Class ---
class InnerWorlds:
    """
    Manages the dream cycle, a state where the AI consolidates and integrates experiences.

    This class orchestrates the retrieval of memory fragments, synthesizes a
    dream scenario, and persists the resulting dream to a journal.
    """
    def __init__(self, dream_journal_path: Path = Path("logs/dream_journal.json")):
        """
        Initializes the InnerWorlds class with a path to the dream journal.
        """
        self.dream_journal_path = dream_journal_path
        self.current_dream = None
        self.last_dream_time = None
        self._load_last_dream_time()
        
    def _load_last_dream_time(self):
        """
        Loads the timestamp of the last dream from the journal file.
        This ensures the schedule_dream_cycle method is persistent across sessions.
        """
        try:
            if self.dream_journal_path.exists():
                with open(self.dream_journal_path, 'r') as file:
                    journal = json.load(file)
                    if journal:
                        last_entry = journal[-1]
                        # Use a fallback for robustness
                        timestamp_str = last_entry.get("timestamp", "1970-01-01T00:00:00Z")
                        self.last_dream_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).timestamp()
        except (FileNotFoundError, json.JSONDecodeError):
            self.last_dream_time = None

    def enter_dream_state(self):
        """
        Initiates a dream cycle for symbolic integration.
        """
        print("[InnerWorlds] Entering dream state...")
        
        # 1. Gather inputs for the dream
        recent_experiences = get_recent_experiences(hours=12)
        core_concepts = retrieve_concepts(limit=random.randint(3, 5))
        unresolved_events = retrieve_events(filter_by="contradiction")
        
        # 2. Generate the dream scenario
        self.current_dream = generate_dream_scenario(core_concepts, recent_experiences, unresolved_events)
        
        # 3. Log the dream
        self._log_dream(self.current_dream)
        
        # 4. Update the last dream time
        self.last_dream_time = time.time()
        
        print(f"[InnerWorlds] Dream cycle complete. Resolved {len(self.current_dream.get('resolved_ideas', []))} ideas.")
        return self.current_dream

    def _log_dream(self, dream_content: dict):
        """
        Persists the generated dream to the dream journal JSON file.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "dream": dream_content,
            "resolved_ideas": dream_content.get("resolutions", []),
            "emotional_shifts": dream_content.get("emotional_movement", {})
        }
        
        journal = []
        # Ensure the log directory exists
        self.dream_journal_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if self.dream_journal_path.exists():
                with open(self.dream_journal_path, 'r') as file:
                    journal = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            print("Warning: Dream journal file not found or corrupted. Starting a new one.")
            journal = []
            
        journal.append(entry)
        
        try:
            with open(self.dream_journal_path, 'w') as file:
                json.dump(journal, file, indent=2)
        except IOError as e:
            print(f"Error: Failed to write to dream journal. {e}")

    def summarize_last_dream(self) -> dict:
        """
        Returns a summary of the most recent dream for reflection or analysis.
        """
        try:
            with open(self.dream_journal_path, 'r') as file:
                journal = json.load(file)
                if journal:
                    return journal[-1]
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        
        return {"error": "No dream data available."}

    def schedule_dream_cycle(self, hours_between: int = 24) -> bool:
        """
        Determines whether a new dream cycle should be triggered based on time elapsed.
        """
        if self.last_dream_time is None:
            return True
        
        time_since_last_dream = time.time() - self.last_dream_time
        return time_since_last_dream >= (hours_between * 3600)

# --- Test Driver ---
if __name__ == "__main__":
    # Clean up the previous log file for a fresh test run
    DREAM_JOURNAL_PATH = Path("logs/dream_journal.json")
    if DREAM_JOURNAL_PATH.exists():
        DREAM_JOURNAL_PATH.unlink()
        print(f"Removed previous dream journal for a clean test run.")

    print("--- Test Run 1: Entering first dream state ---")
    inner_worlds = InnerWorlds()
    dream_scenario = inner_worlds.enter_dream_state()
    print("Generated Dream Scenario:", json.dumps(dream_scenario, indent=2))
    print(f"Dream cycle scheduled? {inner_worlds.schedule_dream_cycle(hours_between=1)}")

    print("\n--- Test Run 2: Summarizing the last dream ---")
    last_dream_summary = inner_worlds.summarize_last_dream()
    print("Last Dream Summary:", json.dumps(last_dream_summary, indent=2))
    
    print("\n--- Test Run 3: Checking schedule before time elapsed ---")
    # Simulate a new session by creating a new instance
    new_inner_worlds = InnerWorlds()
    print(f"Is a new dream cycle due? {new_inner_worlds.schedule_dream_cycle(hours_between=24)}")
    
    print("\n--- Test complete. Please check logs/dream_journal.json for output. ---")
