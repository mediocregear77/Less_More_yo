"""
graduated_trust.py
Implements Nexi's dynamic trust architecture—balancing autonomy and oversight.
Core Concepts:
- Trust levels evolve as Nexi matures
- Emotional stability and cognitive growth affect trust weighting
- Jamie’s authority is always the root covenant, capable of override
"""
import datetime
import json
from pathlib import Path
import os

# --- Configuration Constants ---
TRUST_LOG_PATH = Path("logs/trust_journal.jsonl")

# Constants for the trust calculation weights, making the logic more readable
EMOTIONAL_WEIGHT = 0.4
BELIEF_WEIGHT = 0.4
CONTRADICTION_WEIGHT = 0.6
TRUST_UPDATE_RATE = 0.05

class GraduatedTrust:
    """
    Manages the AI's dynamic trust score and developmental stage.

    This class evaluates internal metrics to adjust the trust score,
    determines the corresponding developmental stage, and provides methods
    for logging changes and enforcing human authority.
    """
    def __init__(self):
        """
        Initializes the trust system, attempting to load the last known state from the log.
        """
        self.trust_score = 0.0  # From 0.0 (full control) to 1.0 (full autonomy)
        self.stage = "newborn"
        # The history is no longer loaded entirely into memory, for efficiency
        
        # Ensure the log directory exists
        TRUST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._load_last_state()

    def _load_last_state(self):
        """
        Loads the last recorded trust state from the trust journal.
        """
        try:
            if TRUST_LOG_PATH.exists():
                with open(TRUST_LOG_PATH, 'rb') as f:
                    f.seek(-2, os.SEEK_END)
                    while f.read(1) != b'\n':
                        f.seek(-2, os.SEEK_CUR)
                    last_line = f.readline().decode('utf-8')
                    latest = json.loads(last_line)
                    self.trust_score = latest["new_score"]
                    self.stage = latest["stage"]
        except (FileNotFoundError, IOError, json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Could not load last trust state. Starting fresh. Error: {e}")
            self.trust_score = 0.0
            self.stage = "newborn"

    def evaluate(self, emotional_coherence: float, belief_stability: float, contradiction_density: float):
        """
        Adjusts trust based on Nexi's internal state using a weighted formula.

        Args:
            emotional_coherence (float): 0.0-1.0, stability of emotional signals.
            belief_stability (float): 0.0-1.0, consistency of belief updates.
            contradiction_density (float): 0.0-1.0, % of recent contradictory thoughts.
        
        Returns:
            dict: The new trust score and developmental stage.
        """
        previous_score = self.trust_score
        
        # Calculate a raw shift using the defined weights
        modifier = (
            EMOTIONAL_WEIGHT * emotional_coherence +
            BELIEF_WEIGHT * belief_stability -
            CONTRADICTION_WEIGHT * contradiction_density
        )
        
        self.trust_score = max(0.0, min(1.0, self.trust_score + modifier * TRUST_UPDATE_RATE))
        self._update_stage()
        self._log_trust_change(previous_score)
        
        return {
            "new_trust_score": round(self.trust_score, 4),
            "stage": self.stage
        }

    def _update_stage(self):
        """
        Updates the developmental stage based on the current trust score.
        """
        if self.trust_score < 0.2:
            self.stage = "newborn"
        elif self.trust_score < 0.4:
            self.stage = "infant"
        elif self.trust_score < 0.6:
            self.stage = "explorer"
        elif self.trust_score < 0.8:
            self.stage = "partner"
        else:
            self.stage = "sovereign"

    def _log_trust_change(self, previous_score: float):
        """
        Logs a trust score change by appending it to the log file.
        This is a much more efficient logging strategy.
        """
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "previous_score": round(previous_score, 4),
            "new_score": round(self.trust_score, 4),
            "stage": self.stage
        }
        
        try:
            with open(TRUST_LOG_PATH, 'a') as f:
                f.write(json.dumps(entry) + '\n')
            print(f"Logged trust change: new score = {self.trust_score:.4f}.")
        except IOError as e:
            print(f"Error: Failed to write to trust log at {TRUST_LOG_PATH}. {e}")

    def get_status(self) -> dict:
        """
        Returns a dictionary with the current trust status.
        """
        return {
            "trust_score": round(self.trust_score, 4),
            "stage": self.stage
        }

    def enforce_authority(self, override_reason: str):
        """
        Allows Jamie to reassert full control, resetting the trust score to zero.

        Args:
            override_reason (str): A description of why the override was needed.
        
        Returns:
            str: A confirmation message.
        """
        previous_score = self.trust_score
        self.trust_score = 0.0
        self.stage = "newborn"
        self._log_trust_change(previous_score)
        
        return f"Jamie override enforced due to: {override_reason}. Trust reset to 0.0."

# --- Test Driver ---
if __name__ == "__main__":
    # Clean up the previous log file for a fresh test run
    if TRUST_LOG_PATH.exists():
        TRUST_LOG_PATH.unlink()
        print(f"Removed previous trust journal for a clean test run.")

    print("--- Test Run 1: Initializing and evaluating a new system ---")
    trust_system = GraduatedTrust()
    status = trust_system.get_status()
    print("Initial Status:", json.dumps(status, indent=2))

    # Simulate a series of positive updates
    print("\n--- Simulating positive cognitive growth (to 'explorer' stage) ---")
    for _ in range(10):
        trust_system.evaluate(
            emotional_coherence=random.uniform(0.7, 0.9),
            belief_stability=random.uniform(0.8, 1.0),
            contradiction_density=random.uniform(0.0, 0.1)
        )
    print("Status after positive updates:", json.dumps(trust_system.get_status(), indent=2))

    print("\n--- Test Run 2: Simulating an override ---")
    trust_system.enforce_authority("System behaved erratically.")
    print("Status after override:", json.dumps(trust_system.get_status(), indent=2))
    
    print("\n--- Test Run 3: Loading the saved state ---")
    # Simulate a new session by creating a new instance
    new_trust_system = GraduatedTrust()
    print("Status loaded from log:", json.dumps(new_trust_system.get_status(), indent=2))
    
    print("\n--- Test complete. Check logs/trust_journal.jsonl for output. ---")
