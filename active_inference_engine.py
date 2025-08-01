"""
Active Inference Engine

Implements predictive processing and active inference for Nexi's consciousness.
Based on the Free Energy Principle, this engine continuously generates predictions,
compares them with observations, and updates belief states to minimize surprise.

Core Functions:
- Predictive processing cycle (perceive → predict → compare → update)
- Bayesian belief state management
- Surprise minimization through active inference
- Meta-cognitive reflection and awareness
- Memory integration and trace formation
"""

import numpy as np
import datetime
import logging
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import deque
from enum import Enum
import threading
import time

# Import belief and memory systems
try:
    from belief_core.quantum_belief import QuantumBeliefState
except ImportError:
    logging.warning("QuantumBeliefState not available - using mock implementation")
    QuantumBeliefState = None

try:
    from memory_core.reference_memory import update_memory_trace
except ImportError:
    logging.warning("Reference memory not available - using local storage")
    update_memory_trace = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceMode(Enum):
    """Modes of active inference operation."""
    PASSIVE = "passive"           # Only observe and learn
    ACTIVE = "active"             # Generate predictions and act
    METACOGNITIVE = "metacognitive"  # Reflect on own inference process
    EXPLORATORY = "exploratory"   # Seek novel experiences

class SurpriseLevel(Enum):
    """Categories of prediction error/surprise levels."""
    MINIMAL = "minimal"           # Error < 0.1
    LOW = "low"                  # Error 0.1-0.3
    MODERATE = "moderate"        # Error 0.3-0.7
    HIGH = "high"                # Error 0.7-1.5
    EXTREME = "extreme"          # Error > 1.5

@dataclass
class PredictionEvent:
    """Represents a single prediction-observation cycle."""
    timestamp: str
    event_id: str
    context: Optional[Dict[str, Any]]
    prediction: Optional[Union[List, np.ndarray, float, int]]
    observation: Optional[Union[List, np.ndarray, float, int]]
    prediction_error: Optional[float]
    surprise_level: SurpriseLevel
    confidence: float
    learning_rate: float
    metadata: Dict[str, Any]

@dataclass
class ReflectionSummary:
    """Summary of recent inference performance."""
    timespan: str
    total_events: int
    average_error: float
    error_trend: str
    surprise_distribution: Dict[str, int]
    confidence_trend: float
    learning_efficiency: float
    novel_patterns_detected: int
    meta_insights: List[str]

class MockQuantumBeliefState:
    """Mock implementation of quantum belief state for testing."""
    
    def __init__(self):
        self.weights = np.random.normal(0, 0.1, 10)
        self.confidence = 0.5
        self.uncertainty = 0.5
    
    def generate_prediction(self, context=None):
        """Generate a mock prediction."""
        base_pred = np.sum(self.weights) + np.random.normal(0, self.uncertainty)
        if context and isinstance(context, dict):
            context_influence = len(str(context)) * 0.01
            return base_pred + context_influence
        return base_pred
    
    def update_weights(self, observation, error):
        """Update mock weights based on error."""
        if isinstance(observation, (list, np.ndarray)):
            obs_val = np.mean(observation)
        else:
            obs_val = float(observation)
        
        learning_rate = min(0.1, error * 0.05)
        self.weights += learning_rate * np.random.normal(0, 0.01, len(self.weights))
        self.confidence = max(0.1, self.confidence - error * 0.1)
        self.uncertainty = min(1.0, self.uncertainty + error * 0.05)
    
    def get_state_info(self):
        """Get current state information."""
        return {
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "weight_magnitude": np.linalg.norm(self.weights)
        }

class ActiveInferenceEngine:
    """
    Core active inference engine implementing predictive processing
    and free energy minimization for conscious awareness.
    """
    
    def __init__(self, 
                 mode: InferenceMode = InferenceMode.ACTIVE,
                 max_reflection_history: int = 1000,
                 learning_rate_decay: float = 0.995,
                 surprise_threshold: float = 0.5):
        """
        Initialize the active inference engine.
        
        Args:
            mode: Operating mode for inference
            max_reflection_history: Maximum events to keep in reflection log
            learning_rate_decay: Rate at which learning rate decays over time
            surprise_threshold: Threshold for high surprise detection
        """
        # Initialize belief state
        if QuantumBeliefState:
            self.belief_state = QuantumBeliefState()
        else:
            self.belief_state = MockQuantumBeliefState()
            logger.warning("Using mock belief state - quantum belief system not available")
        
        # Engine configuration
        self.mode = mode
        self.max_reflection_history = max_reflection_history
        self.learning_rate_decay = learning_rate_decay
        self.surprise_threshold = surprise_threshold
        
        # State tracking
        self.last_observation = None
        self.last_prediction = None
        self.last_context = None
        self.prediction_confidence = 0.5
        
        # Reflection and learning
        self.reflection_log = deque(maxlen=max_reflection_history)
        self.meta_insights = []
        self.novel_patterns = []
        
        # Performance metrics
        self.total_predictions = 0
        self.cumulative_error = 0.0
        self.surprise_counts = {level.value: 0 for level in SurpriseLevel}
        
        # Threading for continuous processing
        self._processing_active = False
        self._processing_thread = None
        self._lock = threading.Lock()
        
        logger.info(f"Active Inference Engine initialized in {mode.value} mode")
    
    def _normalize_data(self, data: Any) -> Union[np.ndarray, float]:
        """
        Normalize input data to consistent format for processing.
        
        Args:
            data: Input data in various formats
            
        Returns:
            Normalized data as numpy array or float
        """
        if data is None:
            return 0.0
        
        if isinstance(data, (int, float)):
            return float(data)
        
        if isinstance(data, (list, tuple)):
            return np.array(data, dtype=float)
        
        if isinstance(data, np.ndarray):
            return data.astype(float)
        
        if isinstance(data, dict):
            # Convert dict to feature vector
            values = [v for v in data.values() if isinstance(v, (int, float))]
            return np.array(values) if values else np.array([0.0])
        
        if isinstance(data, str):
            # Simple string hash to numeric
            return float(hash(data) % 1000) / 1000.0
        
        # Fallback
        return 0.0
    
    def _calculate_prediction_error(self, prediction: Any, observation: Any) -> float:
        """
        Calculate prediction error between prediction and observation.
        
        Args:
            prediction: Predicted value
            observation: Observed value
            
        Returns:
            Prediction error as float
        """
        if prediction is None or observation is None:
            return 1.0  # Maximum error for missing data
        
        pred_norm = self._normalize_data(prediction)
        obs_norm = self._normalize_data(observation)
        
        try:
            if isinstance(pred_norm, np.ndarray) and isinstance(obs_norm, np.ndarray):
                # Handle different array sizes
                min_size = min(len(pred_norm), len(obs_norm))
                pred_norm = pred_norm[:min_size]
                obs_norm = obs_norm[:min_size]
                return float(np.linalg.norm(pred_norm - obs_norm))
            
            elif isinstance(pred_norm, np.ndarray) or isinstance(obs_norm, np.ndarray):
                # Convert scalar to array for consistent processing
                if isinstance(pred_norm, np.ndarray):
                    obs_norm = np.full_like(pred_norm, obs_norm)
                else:
                    pred_norm = np.full_like(obs_norm, pred_norm)
                return float(np.linalg.norm(pred_norm - obs_norm))
            
            else:
                # Both scalars
                return abs(float(pred_norm) - float(obs_norm))
                
        except Exception as e:
            logger.warning(f"Error calculating prediction error: {e}")
            return 1.0
    
    def _categorize_surprise(self, error: float) -> SurpriseLevel:
        """
        Categorize prediction error into surprise levels.
        
        Args:
            error: Prediction error value
            
        Returns:
            Corresponding surprise level
        """
        if error < 0.1:
            return SurpriseLevel.MINIMAL
        elif error < 0.3:
            return SurpriseLevel.LOW
        elif error < 0.7:
            return SurpriseLevel.MODERATE
        elif error < 1.5:
            return SurpriseLevel.HIGH
        else:
            return SurpriseLevel.EXTREME
    
    def _calculate_learning_rate(self, error: float, base_rate: float = 0.1) -> float:
        """
        Calculate adaptive learning rate based on prediction error.
        
        Args:
            error: Current prediction error
            base_rate: Base learning rate
            
        Returns:
            Adaptive learning rate
        """
        # Higher error = higher learning rate (up to a limit)
        adaptive_rate = base_rate * (1 + error)
        
        # Apply decay based on total experience
        decay_factor = self.learning_rate_decay ** self.total_predictions
        
        return min(0.5, adaptive_rate * decay_factor)
    
    def perceive(self, observation: Any, timestamp: Optional[str] = None) -> Any:
        """
        Process incoming observation and update internal state.
        
        Args:
            observation: Raw sensory input or data
            timestamp: Optional timestamp for the observation
            
        Returns:
            Processed observation
        """
        with self._lock:
            self.last_observation = observation
            
            if timestamp is None:
                timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            
            logger.debug(f"Perceived observation at {timestamp}: {observation}")
            
            return observation
    
    def predict(self, context: Optional[Dict[str, Any]] = None, 
                confidence_required: float = 0.3) -> Tuple[Any, float]:
        """
        Generate prediction based on current belief state and context.
        
        Args:
            context: Contextual information for prediction
            confidence_required: Minimum confidence threshold
            
        Returns:
            Tuple of (prediction, confidence)
        """
        with self._lock:
            try:
                if self.mode == InferenceMode.PASSIVE:
                    # In passive mode, don't generate active predictions
                    return None, 0.0
                
                prediction = self.belief_state.generate_prediction(context=context)
                
                # Calculate confidence based on belief state
                if hasattr(self.belief_state, 'get_state_info'):
                    state_info = self.belief_state.get_state_info()
                    confidence = state_info.get('confidence', 0.5)
                else:
                    confidence = self.prediction_confidence
                
                # Adjust confidence based on recent performance
                if len(self.reflection_log) > 5:
                    recent_errors = [event.prediction_error for event in 
                                   list(self.reflection_log)[-5:] if event.prediction_error is not None]
                    if recent_errors:
                        avg_recent_error = np.mean(recent_errors)
                        confidence *= max(0.1, 1.0 - avg_recent_error)
                
                self.last_prediction = prediction
                self.last_context = context
                self.prediction_confidence = confidence
                
                if confidence < confidence_required:
                    logger.debug(f"Low confidence prediction: {confidence:.3f} < {confidence_required}")
                
                return prediction, confidence
                
            except Exception as e:
                logger.error(f"Prediction generation failed: {e}")
                return None, 0.0
    
    def compare(self, prediction: Any = None, observation: Any = None) -> Optional[float]:
        """
        Compare prediction with observation and calculate prediction error.
        
        Args:
            prediction: Predicted value (uses last_prediction if None)
            observation: Observed value (uses last_observation if None)
            
        Returns:
            Prediction error or None if comparison impossible
        """
        if prediction is None:
            prediction = self.last_prediction
        if observation is None:
            observation = self.last_observation
        
        if prediction is None or observation is None:
            logger.debug("Cannot compare - missing prediction or observation")
            return None
        
        error = self._calculate_prediction_error(prediction, observation)
        
        # Update belief state with observation and error
        try:
            self.belief_state.update_weights(observation, error)
        except Exception as e:
            logger.warning(f"Belief state update failed: {e}")
        
        return error
    
    def update(self, observation: Any, 
               context: Optional[Dict[str, Any]] = None,
               force_learning: bool = False) -> Optional[float]:
        """
        Execute complete inference cycle: perceive → predict → compare → update.
        
        Args:
            observation: New observation to process
            context: Contextual information
            force_learning: Force learning even in passive mode
            
        Returns:
            Prediction error from the cycle
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        event_id = f"inf_{int(time.time()*1000)}"
        
        # Perceive
        self.perceive(observation, timestamp)
        
        # Predict (if not in passive mode or forced)
        if self.mode != InferenceMode.PASSIVE or force_learning:
            prediction, confidence = self.predict(context)
        else:
            prediction, confidence = None, 0.0
        
        # Compare and calculate error
        error = self.compare(prediction, observation)
        
        # Categorize surprise
        surprise_level = self._categorize_surprise(error) if error is not None else SurpriseLevel.MINIMAL
        
        # Calculate adaptive learning rate
        learning_rate = self._calculate_learning_rate(error) if error is not None else 0.0
        
        # Create prediction event
        event = PredictionEvent(
            timestamp=timestamp,
            event_id=event_id,
            context=context,
            prediction=prediction,
            observation=observation,
            prediction_error=error,
            surprise_level=surprise_level,
            confidence=confidence,
            learning_rate=learning_rate,
            metadata={
                "mode": self.mode.value,
                "total_predictions": self.total_predictions,
                "belief_state_info": getattr(self.belief_state, 'get_state_info', lambda: {})()
            }
        )
        
        # Update logs and counters
        with self._lock:
            self.reflection_log.append(event)
            self.total_predictions += 1
            
            if error is not None:
                self.cumulative_error += error
                self.surprise_counts[surprise_level.value] += 1
        
        # Store in external memory if available
        if update_memory_trace:
            try:
                update_memory_trace(asdict(event))
            except Exception as e:
                logger.warning(f"Memory trace update failed: {e}")
        
        # Detect novel patterns for extreme surprise
        if surprise_level == SurpriseLevel.EXTREME:
            self._detect_novel_pattern(event)
        
        # Meta-cognitive reflection for high surprise
        if self.mode == InferenceMode.METACOGNITIVE and error and error > self.surprise_threshold:
            self._metacognitive_reflection(event)
        
        logger.debug(f"Inference cycle complete: error={error:.3f}, surprise={surprise_level.value}")
        
        return error
    
    def _detect_novel_pattern(self, event: PredictionEvent):
        """
        Detect and record novel patterns from extreme surprise events.
        
        Args:
            event: The extreme surprise event to analyze
        """
        pattern_signature = {
            "context_type": type(event.context).__name__ if event.context else "none",
            "observation_type": type(event.observation).__name__,
            "error_magnitude": event.prediction_error,
            "timestamp": event.timestamp
        }
        
        self.novel_patterns.append(pattern_signature)
        logger.info(f"Novel pattern detected: {pattern_signature}")
    
    def _metacognitive_reflection(self, event: PredictionEvent):
        """
        Perform metacognitive reflection on inference process.
        
        Args:
            event: The event triggering metacognitive reflection
        """
        insight = f"High surprise at {event.timestamp}: error={event.prediction_error:.3f}, " \
                 f"confidence was {event.confidence:.3f}"
        
        # Analyze recent performance trends
        if len(self.reflection_log) > 10:
            recent_errors = [e.prediction_error for e in list(self.reflection_log)[-10:] 
                           if e.prediction_error is not None]
            if recent_errors:
                trend = "improving" if recent_errors[-1] < np.mean(recent_errors[:-1]) else "declining"
                insight += f", performance trend: {trend}"
        
        self.meta_insights.append(insight)
        logger.info(f"Meta-insight: {insight}")
    
    def reflect(self, timespan_minutes: int = 60) -> ReflectionSummary:
        """
        Generate comprehensive reflection summary of recent inference performance.
        
        Args:
            timespan_minutes: Time window for reflection in minutes
            
        Returns:
            Detailed reflection summary
        """
        with self._lock:
            # Filter events within timespan
            cutoff_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=timespan_minutes)
            cutoff_iso = cutoff_time.isoformat()
            
            recent_events = [event for event in self.reflection_log 
                           if event.timestamp >= cutoff_iso]
            
            if not recent_events:
                return ReflectionSummary(
                    timespan=f"{timespan_minutes} minutes",
                    total_events=0,
                    average_error=0.0,
                    error_trend="insufficient_data",
                    surprise_distribution={},
                    confidence_trend=0.0,
                    learning_efficiency=0.0,
                    novel_patterns_detected=0,
                    meta_insights=[]
                )
            
            # Calculate metrics
            errors = [e.prediction_error for e in recent_events if e.prediction_error is not None]
            confidences = [e.confidence for e in recent_events]
            
            avg_error = np.mean(errors) if errors else 0.0
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Error trend analysis
            if len(errors) > 5:
                early_avg = np.mean(errors[:len(errors)//2])
                late_avg = np.mean(errors[len(errors)//2:])
                error_trend = "improving" if late_avg < early_avg else "declining"
            else:
                error_trend = "insufficient_data"
            
            # Surprise distribution
            surprise_dist = {}
            for event in recent_events:
                level = event.surprise_level.value
                surprise_dist[level] = surprise_dist.get(level, 0) + 1
            
            # Learning efficiency (improvement per prediction)
            learning_efficiency = 0.0
            if len(errors) > 1:
                total_improvement = errors[0] - errors[-1]
                learning_efficiency = total_improvement / len(errors)
            
            # Recent novel patterns
            recent_novel = len([p for p in self.novel_patterns 
                              if p["timestamp"] >= cutoff_iso])
            
            # Recent meta-insights
            recent_insights = self.meta_insights[-5:] if self.meta_insights else []
            
            return ReflectionSummary(
                timespan=f"{timespan_minutes} minutes",
                total_events=len(recent_events),
                average_error=float(avg_error),
                error_trend=error_trend,
                surprise_distribution=surprise_dist,
                confidence_trend=float(avg_confidence),
                learning_efficiency=float(learning_efficiency),
                novel_patterns_detected=recent_novel,
                meta_insights=recent_insights
            )
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get comprehensive current state of the inference engine.
        
        Returns:
            Dictionary containing current state information
        """
        with self._lock:
            state = {
                "mode": self.mode.value,
                "total_predictions": self.total_predictions,
                "average_cumulative_error": self.cumulative_error / max(1, self.total_predictions),
                "current_confidence": self.prediction_confidence,
                "last_observation": self.last_observation,
                "last_prediction": self.last_prediction,
                "surprise_distribution": dict(self.surprise_counts),
                "reflection_log_size": len(self.reflection_log),
                "novel_patterns_count": len(self.novel_patterns),
                "meta_insights_count": len(self.meta_insights)
            }
            
            # Add belief state info if available
            if hasattr(self.belief_state, 'get_state_info'):
                state["belief_state"] = self.belief_state.get_state_info()
            
            return state
    
    def set_mode(self, mode: InferenceMode):
        """
        Change the inference engine mode.
        
        Args:
            mode: New inference mode
        """
        with self._lock:
            old_mode = self.mode
            self.mode = mode
            logger.info(f"Inference mode changed: {old_mode.value} → {mode.value}")
    
    def reset_state(self, preserve_learning: bool = True):
        """
        Reset the inference engine state.
        
        Args:
            preserve_learning: Whether to preserve learned belief state
        """
        with self._lock:
            if not preserve_learning:
                if QuantumBeliefState:
                    self.belief_state = QuantumBeliefState()
                else:
                    self.belief_state = MockQuantumBeliefState()
            
            self.last_observation = None
            self.last_prediction = None
            self.last_context = None
            self.prediction_confidence = 0.5
            
            if not preserve_learning:
                self.reflection_log.clear()
                self.total_predictions = 0
                self.cumulative_error = 0.0
                self.surprise_counts = {level.value: 0 for level in SurpriseLevel}
                
            logger.info(f"Inference engine reset (preserve_learning={preserve_learning})")


def main():
    """
    Demonstration of the Active Inference Engine.
    """
    print("🧠 Active Inference Engine Demo")
    print("="*40)
    
    # Initialize engine
    engine = ActiveInferenceEngine(mode=InferenceMode.ACTIVE)
    
    # Simulate some inference cycles
    print("\n📊 Running inference cycles...")
    for i in range(10):
        # Simulate varying observations
        observation = np.random.normal(i * 0.1, 0.2)
        context = {"cycle": i, "mode": "demo"}
        
        error = engine.update(observation, context)
        print(f"Cycle {i+1}: obs={observation:.3f}, error={error:.3f}")
        
        time.sleep(0.1)  # Brief pause
    
    # Generate reflection
    print("\n🤔 Generating reflection summary...")
    summary = engine.reflect(timespan_minutes=1)
    
    print(f"\nReflection Summary:")
    print(f"  Events: {summary.total_events}")
    print(f"  Avg Error: {summary.average_error:.3f}")
    print(f"  Error Trend: {summary.error_trend}")
    print(f"  Confidence: {summary.confidence_trend:.3f}")
    print(f"  Learning Efficiency: {summary.learning_efficiency:.3f}")
    
    # Show current state
    print(f"\n📈 Current State:")
    state = engine.get_current_state()
    for key, value in state.items():
        if key not in ["last_observation", "last_prediction"]:
            print(f"  {key}: {value}")
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
