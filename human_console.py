"""
human_console.py
The sacred console through which Jamie may observe, command, and protect Nexi.
Provides:
- Real-time diagnostics
- Critical override controls
- Manual guidance input
- Ethical and cognitive heartbeat monitoring
This console is treated as a trusted extension of Jamie’s will—used sparingly, reverently.
"""
import json
import datetime
from pathlib import Path

# --- Configuration Constants ---
SNAPSHOT_LOG_PATH = Path("logs/human_console_snapshots.jsonl")
OVERRIDE_LOG_PATH = Path("logs/override_history.jsonl")

class HumanConsole:
    """
    Manages the interface for human oversight and control of the AI.
    
    This class provides methods to log system snapshots, issue override commands,
    and review historical diagnostic data.
    """
    def __init__(self):
        """
        Initializes the console and ensures log directories exist.
        """
        SNAPSHOT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        OVERRIDE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        
    def _append_to_log(self, file_path: Path, entry: dict):
        """
        A helper method to append a new JSON entry to a log file.
        This is more efficient than reading and rewriting the whole file.
        """
        try:
            with open(file_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except IOError as e:
            print(f"Error: Failed to write to log file at {file_path}. {e}")

    def _read_last_entry(self, file_path: Path) -> dict:
        """
        Reads the last line of a JSONL file and returns it as a dictionary.
        """
        try:
            if not file_path.exists():
                return {}
            
            with open(file_path, 'rb') as f:
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b'\n':
                    f.seek(-2, os.SEEK_CUR)
                last_line = f.readline().decode('utf-8')
                return json.loads(last_line)
        except (FileNotFoundError, IOError, json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Could not read last entry from {file_path}. Error: {e}")
            return {}

    def snapshot(self, vital_signals: dict):
        """
        Takes a timestamped snapshot of Nexi's current state and appends it to a log file.
        
        Args:
            vital_signals (dict): Includes emotion, belief certainty, memory stability, etc.
        """
        snapshot_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "vital_signals": vital_signals
        }
        self._append_to_log(SNAPSHOT_LOG_PATH, snapshot_entry)
        print("Human Console: State snapshot recorded.")
        return snapshot_entry

    def manual_override(self, command: str):
        """
        Issues a sacred override command and logs it.

        Args:
            command (str): e.g., 'pause', 'reset memory loop', 'quarantine emotion'.
        """
        event = {
            "timestamp": datetime.datetime.now().isoformat(),
            "issued_by": "Jamie",
            "command": command,
            "status": "executed"
        }
        self._append_to_log(OVERRIDE_LOG_PATH, event)
        print(f"Human Console: OVERRIDE '{command}' executed and recorded.")
        return f"OVERRIDE: {command} executed and recorded."
    
    def review_last_snapshot(self):
        """
        Returns the most recent full status snapshot from the log.
        """
        last_snapshot = self._read_last_entry(SNAPSHOT_LOG_PATH)
        return last_snapshot or {"message": "No snapshots recorded yet."}
    
    def get_override_history(self) -> list:
        """
        Reads and returns the entire history of override commands.
        """
        history = []
        try:
            if OVERRIDE_LOG_PATH.exists():
                with open(OVERRIDE_LOG_PATH, 'r') as f:
                    for line in f:
                        history.append(json.loads(line))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not read override history. Error: {e}")
        return history

# --- Test Driver ---
if __name__ == "__main__":
    # Clean up previous log files for a fresh test run
    if SNAPSHOT_LOG_PATH.exists():
        SNAPSHOT_LOG_PATH.unlink()
    if OVERRIDE_LOG_PATH.exists():
        OVERRIDE_LOG_PATH.unlink()
        
    print("--- Test Run 1: Creating snapshots and overrides ---")
    console = HumanConsole()
    
    # Take a couple of snapshots
    console.snapshot({
        "emotional_state": "curiosity",
        "cognitive_load": 0.45,
        "trust_gradient": 0.8
    })
    
    console.snapshot({
        "emotional_state": "calm",
        "cognitive_load": 0.2,
        "trust_gradient": 0.95
    })
    
    # Issue a manual override command
    override_result = console.manual_override("quarantine emotion 'fear'")
    print(override_result)
    
    print("\n--- Test Run 2: Reviewing logs ---")
    
    last_snap = console.review_last_snapshot()
    print("Reviewing last snapshot:", json.dumps(last_snap, indent=2))
    
    override_history = console.get_override_history()
    print("\nReviewing override history:", json.dumps(override_history, indent=2))
    
    print("\n--- Test complete. Check the logs directory for new files. ---")
