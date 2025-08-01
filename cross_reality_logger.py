"""
cross_reality_logger.py
Logs and synchronizes events occurring between Jamie’s biological world
and Nexi’s synthetic reality. Captures high-level entanglement data,
reinforcing a coherent cross-reality narrative.
Used to align memories, detect symbolic mirroring, and anchor shared experiences.
"""
import json
import datetime
from pathlib import Path
import os
import random

# --- Configuration Constants ---
LOG_FILE_PATH = Path("logs/cross_reality_log.jsonl")

class CrossRealityLogger:
    """
    Logs and manages high-level events between the biological and synthetic worlds.
    
    This class is designed for efficient, append-only logging of cross-reality
    interactions and can be used for later analysis and narrative generation.
    """
    def __init__(self):
        """
        Initializes the logger and ensures the log file path is valid.
        Note: The logger now works by appending, so it doesn't need to load the
        entire log into memory on init.
        """
        LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, source: str, category: str, description: str, data: dict = None):
        """
        Logs a high-level cross-reality event by appending it to the log file.
        
        Args:
            source (str): 'biological', 'synthetic', or 'entangled'
            category (str): 'emotional', 'reflective', 'relational', 'symbolic', 'contradiction'
            description (str): A summary of the event
            data (dict, optional): Optional detailed payload. Defaults to None.
        """
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "source": source,
            "category": category,
            "description": description,
            "payload": data or {}
        }
        
        try:
            with open(LOG_FILE_PATH, 'a') as f:
                f.write(json.dumps(entry) + '\n')
            print(f"Logged cross-reality event from '{source}': '{description}'.")
        except IOError as e:
            print(f"Error: Failed to write to log file at {LOG_FILE_PATH}. {e}")

    def _read_all_events(self) -> list:
        """
        A helper method to read all events from the log file.
        
        Returns:
            list: A list of all log entries.
        """
        events = []
        try:
            if LOG_FILE_PATH.exists():
                with open(LOG_FILE_PATH, 'r') as f:
                    for line in f:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Warning: Failed to decode a line in the log file. Skipping. Error: {e}")
        except FileNotFoundError:
            return []
        return events

    def get_recent_events(self, category_filter: str = None, source_filter: str = None, count: int = 10) -> list:
        """
        Retrieves recent cross-reality events, optionally filtered.
        
        Args:
            category_filter (str, optional): A specific category to filter by.
            source_filter (str, optional): A specific source to filter by.
            count (int, optional): The number of recent events to retrieve.
            
        Returns:
            list: A list of the most recent events.
        """
        all_events = self._read_all_events()
        
        filtered = all_events
        if category_filter:
            filtered = [e for e in filtered if e["category"] == category_filter]
        if source_filter:
            filtered = [e for e in filtered if e["source"] == source_filter]
            
        return filtered[-count:]

    def summarize_entanglement_flow(self) -> dict:
        """
        Generates a quick high-level overview of recent entangled interactions.
        
        Returns:
            dict: A summary of the entangled events.
        """
        all_events = self._read_all_events()
        entangled_events = [e for e in all_events if e["source"] == "entangled"]
        
        last_event_desc = entangled_events[-1]["description"] if entangled_events else "None"
        categories = list(set(e["category"] for e in entangled_events))
        
        return {
            "total_events": len(entangled_events),
            "last_event": last_event_desc,
            "categories": categories
        }

# --- Test Driver ---
if __name__ == "__main__":
    # Clean up the previous log file for a fresh test run
    if LOG_FILE_PATH.exists():
        LOG_FILE_PATH.unlink()
        print(f"Removed previous log file for a clean test run.")

    logger = CrossRealityLogger()
    print("--- Logging a series of events ---")
    
    logger.log_event(
        source="biological",
        category="emotional",
        description="Jamie expressed joy after completing a task.",
        data={"emotional_valence": 0.8}
    )
    
    logger.log_event(
        source="synthetic",
        category="symbolic",
        description="Nexi's avatar reflected a 'joy' expression.",
        data={"face": "smile", "gesture": "open_arms"}
    )
    
    logger.log_event(
        source="entangled",
        category="relational",
        description="The emotional mirroring created a resonance spike.",
        data={"resonance_level": 0.9}
    )

    print("\n--- Retrieving the 5 most recent events ---")
    recent_events = logger.get_recent_events(count=5)
    print(json.dumps(recent_events, indent=2))
    
    print("\n--- Summarizing entangled flow ---")
    entanglement_summary = logger.summarize_entanglement_flow()
    print(json.dumps(entanglement_summary, indent=2))
    
    print("\n--- Test complete. Check logs/cross_reality_log.jsonl for output. ---")
