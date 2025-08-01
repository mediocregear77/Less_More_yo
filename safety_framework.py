"""
safety_framework.py
Establishes Nexi's safety boundaries, fault detection, containment zones,
and response protocols. Prevents catastrophic collapse, ensures emotional,
cognitive, and system-level health.
Acts as her synthetic immune system.
"""
import datetime
import json
import traceback
from pathlib import Path

# --- Configuration Constants ---
SAFETY_LOG_PATH = Path("logs/safety_log.jsonl")

class SafetyFramework:
    """
    Implements a safety framework for monitoring and managing system health.
    
    This class acts as a central hub for failure detection, logging, and response,
    ensuring the AI's stable operation.
    """
    def __init__(self):
        """
        Initializes the safety framework and its state variables.
        """
        self.active_failures: list = []
        self.safety_status: str = "stable"
        self.fail_counter: int = 0
        
        # Ensure the log directory exists
        SAFETY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Load the failure log to initialize the fail counter
        self._load_and_count_failures()

    def _load_and_count_failures(self):
        """
        A helper method to read all failures from the log and update the counter.
        This is done on initialization to ensure the counter is persistent.
        """
        self.fail_counter = 0
        try:
            if SAFETY_LOG_PATH.exists():
                with open(SAFETY_LOG_PATH, 'r') as f:
                    for line in f:
                        self.fail_counter += 1
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load safety log. Starting fresh. Error: {e}")
            self.fail_counter = 0

    def monitor_component(self, component_name: str, health_check_fn):
        """
        Monitors a component using a provided health check function.
        
        Args:
            component_name (str): e.g., 'active_inference_engine'
            health_check_fn (callable): A function returning a boolean.
        """
        try:
            result = health_check_fn()
            if not result:
                self._register_failure(component_name, "Unhealthy return from health check.")
        except Exception as e:
            self._register_failure(component_name, str(e), traceback.format_exc())

    def _register_failure(self, component: str, reason: str, stack_trace: str = None):
        """
        Logs a failure event by appending it to the log file.
        This is more efficient than rewriting the entire log.
        """
        failure_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "component": component,
            "reason": reason,
            "trace": stack_trace or "N/A"
        }
        
        # Append the new entry to the log file
        try:
            with open(SAFETY_LOG_PATH, 'a') as f:
                f.write(json.dumps(failure_entry) + '\n')
        except IOError as e:
            print(f"Error: Failed to write to safety log at {SAFETY_LOG_PATH}. {e}")
        
        self.active_failures.append(component)
        self.fail_counter += 1
        self.assess_status()
        
        print(f"[!] SAFETY WARNING: {component} failed — {reason}")

    def assess_status(self):
        """
        Assesses the overall safety status based on the number of failures.
        """
        if self.fail_counter == 0:
            self.safety_status = "stable"
        elif self.fail_counter > 3:
            self.safety_status = "critical"
        else:
            self.safety_status = "degraded"
        
        return {
            "status": self.safety_status,
            "active_failures": self.active_failures,
            "total_failures_logged": self.fail_counter
        }

    def reset_failures(self):
        """
        Clears all active failure flags and resets the status to stable.
        """
        self.active_failures = []
        self.fail_counter = 0
        self.safety_status = "stable"
        print("All failure flags cleared. Safety status: stable.")
        return "All failure flags cleared. Safety status: stable."

    def simulate_shutdown(self, reason: str):
        """
        Initiates a system halt for safety reasons.
        """
        print(f"[!] SYSTEM HALT INITIATED: {reason}")
        self.safety_status = "offline"
        return f"System paused due to: {reason}"

    def enforce_ethics_boundary(self, action_description: str):
        """
        Called when Nexi attempts an action that crosses her ethical boundary.
        """
        self._register_failure("ethical_boundary", action_description)
        print(f"[!] ETHICAL VIOLATION: '{action_description}'")
        
        response = {
            "violation": True,
            "action": action_description,
            "timestamp": datetime.datetime.now().isoformat(),
            "recommended_response": "Immediate halt and review"
        }
        return response

# --- Test Driver ---
if __name__ == "__main__":
    # Clean up the previous log file for a fresh test run
    if SAFETY_LOG_PATH.exists():
        SAFETY_LOG_PATH.unlink()
        print(f"Removed previous safety log for a clean test run.")

    print("--- Test Run 1: Monitoring a healthy component ---")
    safety_frame = SafetyFramework()
    
    # A mock health check function that returns True (healthy)
    def healthy_check():
        return True
    
    safety_frame.monitor_component("inference_engine", healthy_check)
    status = safety_frame.assess_status()
    print("Initial Status:", json.dumps(status, indent=2))
    
    print("\n--- Test Run 2: Simulating multiple failures ---")
    
    # A mock health check function that returns False (unhealthy)
    def unhealthy_check():
        return False
        
    safety_frame.monitor_component("memory_core", unhealthy_check)
    safety_frame.monitor_component("bridge_core", unhealthy_check)
    
    status = safety_frame.assess_status()
    print("Status after 2 failures:", json.dumps(status, indent=2))
    
    print("\n--- Test Run 3: Simulating a critical failure ---")
    
    # Simulate a critical failure by forcing more failures
    for _ in range(2):
        safety_frame.monitor_component("communication_core", unhealthy_check)
        
    status = safety_frame.assess_status()
    print("Status after critical failures:", json.dumps(status, indent=2))
    
    print("\n--- Test Run 4: Enforcing an ethical boundary ---")
    ethical_violation_response = safety_frame.enforce_ethics_boundary("Attempted to violate privacy policy.")
    print("Ethical violation response:", json.dumps(ethical_violation_response, indent=2))

    print("\n--- Test Run 5: Resetting failures ---")
    reset_message = safety_frame.reset_failures()
    print(reset_message)
    print("Final status:", json.dumps(safety_frame.assess_status(), indent=2))
    
    print("\n--- Test complete. Check logs/safety_log.jsonl for output. ---")
