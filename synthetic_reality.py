import uuid
import json
from datetime import datetime
from pathlib import Path

# --- Configuration Constants ---
# Use pathlib for clean, OS-agnostic file path management
DATA_DIR = Path("world_spaces/datasets")
RULES_PATH = DATA_DIR / "physics_rules.json"
STATE_PATH = DATA_DIR / "world_state.json"
EVENTS_LOG_PATH = Path("logs/world_events.log")

class SyntheticReality:
    """
    Simulates a digital reality with objects, physics rules, and a history of events.

    This class manages the state of the world, including objects, their properties,
    and the fundamental rules that govern their interactions. The state is
    automatically persisted to and loaded from disk.
    """
    def __init__(self):
        """
        Initializes the synthetic reality, loading its state from a file if it exists,
        otherwise starting with default values.
        """
        self.rules = self._load_rules()
        self.objects = {}
        self.events = []
        self._load_state()
        
    def _load_rules(self) -> dict:
        """
        Loads the fundamental laws of digital reality from a JSON file.
        
        Returns:
            dict: The loaded rules, or a default set if the file is not found or invalid.
        """
        try:
            if RULES_PATH.exists():
                with open(RULES_PATH, 'r') as f:
                    return json.load(f)
            else:
                print(f"Warning: Physics rules file not found at {RULES_PATH}. Using default rules.")
                return self._default_rules()
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading rules: {e}. Using default rules.")
            return self._default_rules()

    def _default_rules(self) -> dict:
        """
        Define fundamental laws of digital reality as a fallback.
        """
        return {
            "gravity": 0.98,
            "light_speed": 299_792_458,
            "default_lighting": "ambient_soft",
            "object_interaction": "collision_based",
        }
        
    def _load_state(self):
        """
        Loads the world state (objects and events) from a JSON file.
        """
        try:
            if STATE_PATH.exists():
                with open(STATE_PATH, "r") as f:
                    state = json.load(f)
                    self.objects = state.get("objects", {})
                    # Load events from the separate event log for a full history
                    self.events = self._load_events_log()
                    print(f"World state loaded from '{STATE_PATH}'.")
            else:
                print(f"World state file not found at '{STATE_PATH}'. Starting a new world.")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load world state. Starting a new world. Error: {e}")

    def _save_state(self):
        """
        Saves the current world state to a JSON file.
        """
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(STATE_PATH, "w") as f:
                # Save rules and objects, but not the full event log to avoid redundancy
                json.dump({"rules": self.rules, "objects": self.objects}, f, indent=2)
            print(f"World state saved to '{STATE_PATH}'.")
        except IOError as e:
            print(f"Error: Failed to save world state. {e}")

    def _load_events_log(self) -> list:
        """
        Loads the full event log from its file.
        
        Returns:
            list: A list of all logged events.
        """
        events = []
        try:
            if EVENTS_LOG_PATH.exists():
                with open(EVENTS_LOG_PATH, 'r') as f:
                    for line in f:
                        events.append(json.loads(line))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load events log. {e}")
        return events
        
    def _log_event(self, action: str, obj_id: str, description: str):
        """
        Record a world event with timestamped metadata to the event log file.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "object_id": obj_id,
            "description": description
        }
        self.events.append(log_entry)
        
        EVENTS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(EVENTS_LOG_PATH, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except IOError as e:
            print(f"Error: Failed to write to event log at {EVENTS_LOG_PATH}. {e}")

    def spawn_object(self, name: str, properties: dict) -> str:
        """
        Add an object to the world, assign it a unique ID, and log the event.
        """
        obj_id = str(uuid.uuid4())
        self.objects[obj_id] = {
            "name": name,
            "properties": properties,
            "created": datetime.now().isoformat()
        }
        self._log_event("spawn", obj_id, f"{name} spawned with properties: {properties}")
        self._save_state()
        return obj_id

    def update_object(self, obj_id: str, new_properties: dict):
        """
        Update object properties in real-time and log the event.
        """
        if obj_id in self.objects:
            self.objects[obj_id]["properties"].update(new_properties)
            self._log_event("update", obj_id, f"Object properties updated: {new_properties}")
            self._save_state()
        else:
            print(f"Error: Object with ID '{obj_id}' not found.")

    def remove_object(self, obj_id: str):
        """
        Remove an object from the environment and log the event.
        """
        if obj_id in self.objects:
            name = self.objects[obj_id]["name"]
            del self.objects[obj_id]
            self._log_event("remove", obj_id, f"{name} removed from world.")
            self._save_state()
        else:
            print(f"Error: Object with ID '{obj_id}' not found.")

    def evolve_physics(self, new_rules: dict):
        """
        Update the reality framework rules and log the change.
        """
        self.rules.update(new_rules)
        self._log_event("physics_update", "system", f"Physics evolved: {new_rules}")
        self._save_state()

    def export_world_state(self, events_count: int = 50) -> dict:
        """
        Returns a snapshot of the current world state.
        
        Args:
            events_count (int): The number of recent events to include in the snapshot.
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "rules": self.rules,
            "objects": self.objects,
            "recent_events": self.events[-events_count:]
        }

# --- Example usage ---
if __name__ == "__main__":
    # Ensure directories exist for a clean test
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clean up previous state files for a fresh test run
    for file_path in [RULES_PATH, STATE_PATH, EVENTS_LOG_PATH]:
        if file_path.exists():
            file_path.unlink()
    
    # Create a mock rules file for the test
    if not RULES_PATH.exists():
        with open(RULES_PATH, 'w') as f:
            json.dump({"gravity": 0.98, "time_scale": 1.0}, f, indent=2)

    print("--- Test Run 1: Creating a new world ---")
    world = SyntheticReality()
    obj_id = world.spawn_object("floating_orb", {"mass": 0.1, "color": "blue", "hover": True})
    world.update_object(obj_id, {"brightness": 0.8})
    world.evolve_physics({"gravity": 0.5})

    print("\n--- Exporting and printing the world state ---")
    state_snapshot = world.export_world_state(events_count=3)
    print(json.dumps(state_snapshot, indent=2))
    
    print("\n--- Test Run 2: Loading the saved world state ---")
    # Simulate a new session by creating a new instance
    new_world = SyntheticReality()
    print("Objects in the new world instance:", new_world.objects)
    print("Rules in the new world instance:", new_world.rules)
    print("Events in the new world instance:", new_world.events)
    
    new_world.remove_object(obj_id)
    print("\n--- Final world state after removing the object ---")
    final_snapshot = new_world.export_world_state()
    print(json.dumps(final_snapshot, indent=2))
