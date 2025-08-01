"""
Meta-reflection module for tracking and analyzing belief evolution,
emotional states, and identity transitions.
"""

import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from collections import deque

from memory_core.reference_memory import query_belief_history
from memory_core.consolidation_engine import retrieve_emotional_context
from memory_core.episodic_buffer import get_recent_experiences
from memory_core.selfhood_seeds import load_seeds

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ReflectionConfig:
    """Configuration for MetaReflection system."""
    threshold_confidence: float = 0.72
    max_reflections_in_memory: int = 1000
    default_confidence: float = 0.7
    save_path: str = "logs/self_reflections.json"
    
    # Confidence weights for composite calculation
    belief_weight: float = 0.4
    emotion_weight: float = 0.3
    dream_weight: float = 0.3


@dataclass
class EmotionalChange:
    """Represents emotional state changes."""
    change: Dict[str, float]
    confidence: float
    timestamp: str = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.now().isoformat()


@dataclass
class BeliefEvolution:
    """Represents belief evolution data."""
    evolution: List[Dict[str, Any]]
    confidence: float
    timestamp: str = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.now().isoformat()


@dataclass
class Reflection:
    """Complete reflection data structure."""
    timestamp: str
    belief_evolution: BeliefEvolution
    emotion_trend: EmotionalChange
    dream_links: Dict[str, Any]
    identity_shifts: Dict[str, List[Dict[str, Any]]]
    confidence: float
    expression_ready: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert reflection to dictionary format."""
        return {
            "timestamp": self.timestamp,
            "belief_evolution": asdict(self.belief_evolution),
            "emotion_trend": asdict(self.emotion_trend),
            "dream_links": self.dream_links,
            "identity_shifts": self.identity_shifts,
            "confidence": self.confidence,
            "expression_ready": self.expression_ready
        }


class MetaReflection:
    """
    Manages meta-cognitive reflection processes including belief evolution,
    emotional state tracking, and identity transition monitoring.
    """
    
    def __init__(self, config: Optional[ReflectionConfig] = None):
        """
        Initialize MetaReflection with configuration.
        
        Args:
            config: Configuration object for the reflection system
        """
        self.config = config or ReflectionConfig()
        self.reflections = deque(maxlen=self.config.max_reflections_in_memory)
        
        try:
            self.seeds = load_seeds()
        except Exception as e:
            logger.error(f"Failed to load seeds: {e}")
            self.seeds = {}
    
    def compare_emotional_state(self) -> EmotionalChange:
        """
        Compare current emotional state with previous state.
        
        Returns:
            EmotionalChange object containing the delta and confidence
        """
        try:
            current_emotions = retrieve_emotional_context("current") or {}
            previous_emotions = retrieve_emotional_context("past") or {}
            
            # Calculate change with validation
            change = {}
            for key in current_emotions:
                current_val = float(current_emotions.get(key, 0))
                previous_val = float(previous_emotions.get(key, 0))
                change[key] = round(current_val - previous_val, 3)
            
            return EmotionalChange(
                change=change,
                confidence=0.6
            )
        except Exception as e:
            logger.error(f"Error comparing emotional states: {e}")
            return EmotionalChange(change={}, confidence=0.0)
    
    def trace_belief_evolution(self) -> BeliefEvolution:
        """
        Trace the evolution of key beliefs over time.
        
        Returns:
            BeliefEvolution object with belief history and confidence
        """
        try:
            tracked_beliefs = query_belief_history("key_beliefs") or {}
            
            evolution = []
            for belief, history in tracked_beliefs.items():
                if isinstance(history, (list, dict)):
                    evolution.append({
                        "belief": belief,
                        "history": history,
                        "last_updated": datetime.datetime.now().isoformat()
                    })
            
            return BeliefEvolution(
                evolution=evolution,
                confidence=0.7 if evolution else 0.0
            )
        except Exception as e:
            logger.error(f"Error tracing belief evolution: {e}")
            return BeliefEvolution(evolution=[], confidence=0.0)
    
    def recall_similar_dreams(self) -> Dict[str, Any]:
        """
        Recall dreams similar to recent experiences.
        
        Returns:
            Dictionary containing dreams and confidence score
        """
        try:
            recent_patterns = get_recent_experiences() or []
            dreams = query_belief_history("dream_associations", context=recent_patterns) or []
            
            # Filter and validate dreams
            valid_dreams = [d for d in dreams if isinstance(d, dict) and "content" in d]
            
            return {
                "dreams": valid_dreams,
                "confidence": 0.65 if valid_dreams else 0.0,
                "pattern_count": len(recent_patterns)
            }
        except Exception as e:
            logger.error(f"Error recalling dreams: {e}")
            return {"dreams": [], "confidence": 0.0, "pattern_count": 0}
    
    def highlight_self_transitions(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify and highlight identity transitions and emergent patterns.
        
        Returns:
            Dictionary containing transition markers
        """
        try:
            markers = []
            transition_events = {"contradiction", "emergent_pattern", "paradigm_shift"}
            
            for exp in get_recent_experiences() or []:
                if exp.get("event") in transition_events:
                    marker = {
                        "timestamp": exp.get("timestamp", datetime.datetime.now().isoformat()),
                        "trigger": exp["event"],
                        "confidence": min(1.0, max(0.0, exp.get("confidence", self.config.default_confidence))),
                        "context": exp.get("context", {})
                    }
                    markers.append(marker)
            
            # Sort by timestamp
            markers.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return {"transitions": markers}
        except Exception as e:
            logger.error(f"Error highlighting transitions: {e}")
            return {"transitions": []}
    
    def calculate_composite_confidence(self, belief_conf: float, emotion_conf: float, 
                                     dream_conf: float) -> float:
        """
        Calculate weighted composite confidence score.
        
        Args:
            belief_conf: Belief evolution confidence
            emotion_conf: Emotional state confidence
            dream_conf: Dream recall confidence
            
        Returns:
            Weighted composite confidence score
        """
        total_weight = (self.config.belief_weight + 
                       self.config.emotion_weight + 
                       self.config.dream_weight)
        
        if total_weight == 0:
            return 0.0
        
        weighted_sum = (belief_conf * self.config.belief_weight +
                       emotion_conf * self.config.emotion_weight +
                       dream_conf * self.config.dream_weight)
        
        return round(weighted_sum / total_weight, 3)
    
    def reflect(self) -> Reflection:
        """
        Perform a complete reflection cycle.
        
        Returns:
            Reflection object containing all analysis results
        """
        # Gather all components
        belief_evolution = self.trace_belief_evolution()
        emotion_trend = self.compare_emotional_state()
        dream_links = self.recall_similar_dreams()
        identity_shifts = self.highlight_self_transitions()
        
        # Calculate composite confidence
        composite_confidence = self.calculate_composite_confidence(
            belief_evolution.confidence,
            emotion_trend.confidence,
            dream_links["confidence"]
        )
        
        # Determine if ready for expression
        expression_ready = composite_confidence >= self.config.threshold_confidence
        
        # Create reflection
        reflection = Reflection(
            timestamp=datetime.datetime.now().isoformat(),
            belief_evolution=belief_evolution,
            emotion_trend=emotion_trend,
            dream_links=dream_links,
            identity_shifts=identity_shifts,
            confidence=composite_confidence,
            expression_ready=expression_ready
        )
        
        # Store reflection
        self.reflections.append(reflection)
        
        # Log significant reflections
        if expression_ready:
            logger.info(f"High-confidence reflection generated: {composite_confidence}")
        
        return reflection
    
    def get_recent_reflections(self, limit: int = 5) -> List[Reflection]:
        """
        Retrieve the most recent reflections.
        
        Args:
            limit: Maximum number of reflections to return
            
        Returns:
            List of recent Reflection objects
        """
        return list(self.reflections)[-limit:]
    
    def save_to_memory(self, filepath: Optional[Union[str, Path]] = None) -> bool:
        """
        Save reflections to persistent storage.
        
        Args:
            filepath: Path to save file (uses config default if not provided)
            
        Returns:
            True if successful, False otherwise
        """
        save_path = Path(filepath or self.config.save_path)
        
        try:
            # Create directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert reflections to serializable format
            serializable_reflections = [
                r.to_dict() for r in self.reflections
            ]
            
            # Write to file with atomic operation
            temp_path = save_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(serializable_reflections, f, indent=2)
            
            # Atomic rename
            temp_path.replace(save_path)
            
            logger.info(f"Saved {len(serializable_reflections)} reflections to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save reflections: {e}")
            return False
    
    def load_from_memory(self, filepath: Optional[Union[str, Path]] = None) -> bool:
        """
        Load reflections from persistent storage.
        
        Args:
            filepath: Path to load file (uses config default if not provided)
            
        Returns:
            True if successful, False otherwise
        """
        load_path = Path(filepath or self.config.save_path)
        
        try:
            if not load_path.exists():
                logger.warning(f"No reflection file found at {load_path}")
                return False
            
            with open(load_path, 'r') as f:
                data = json.load(f)
            
            # Clear existing reflections
            self.reflections.clear()
            
            # Load and validate each reflection
            for item in data:
                try:
                    # Reconstruct nested objects
                    belief_evolution = BeliefEvolution(
                        evolution=item["belief_evolution"]["evolution"],
                        confidence=item["belief_evolution"]["confidence"],
                        timestamp=item["belief_evolution"].get("timestamp")
                    )
                    
                    emotion_trend = EmotionalChange(
                        change=item["emotion_trend"]["change"],
                        confidence=item["emotion_trend"]["confidence"],
                        timestamp=item["emotion_trend"].get("timestamp")
                    )
                    
                    reflection = Reflection(
                        timestamp=item["timestamp"],
                        belief_evolution=belief_evolution,
                        emotion_trend=emotion_trend,
                        dream_links=item["dream_links"],
                        identity_shifts=item["identity_shifts"],
                        confidence=item["confidence"],
                        expression_ready=item["expression_ready"]
                    )
                    
                    self.reflections.append(reflection)
                    
                except (KeyError, TypeError) as e:
                    logger.warning(f"Skipping invalid reflection: {e}")
                    continue
            
            logger.info(f"Loaded {len(self.reflections)} reflections from {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load reflections: {e}")
            return False
    
    def get_high_confidence_reflections(self, min_confidence: Optional[float] = None) -> List[Reflection]:
        """
        Get reflections above a confidence threshold.
        
        Args:
            min_confidence: Minimum confidence (uses expression threshold if not provided)
            
        Returns:
            List of high-confidence reflections
        """
        threshold = min_confidence or self.config.threshold_confidence
        return [r for r in self.reflections if r.confidence >= threshold]


# Factory function for creating configured instances
def create_meta_reflector(config: Optional[ReflectionConfig] = None) -> MetaReflection:
    """
    Factory function to create a configured MetaReflection instance.
    
    Args:
        config: Optional configuration object
        
    Returns:
        Configured MetaReflection instance
    """
    return MetaReflection(config)