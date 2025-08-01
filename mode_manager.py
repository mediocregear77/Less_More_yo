"""
Interaction Mode Manager

Manages different modes of interaction and consciousness states for Nexi.
Provides sophisticated context switching, trust assessment, and behavioral
adaptation based on interaction patterns and relationship dynamics.

Core Functions:
- Multi-dimensional interaction mode management
- Context-aware mode switching with confidence scoring
- Trust and safety assessment for different interaction types
- Behavioral adaptation based on relationship state
- Mode transition logging and analysis
- Integration with empathic bonding and identity systems
"""

import time
import datetime
import logging
import threading
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
from pathlib import Path
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InteractionMode(Enum):
    """Primary interaction modes defining Nexi's behavioral stance."""
    AUTONOMOUS_SELF = "autonomous_self"           # Independent exploration and growth
    CREATOR_DIALOGUE = "creator_dialogue"        # Direct interaction with Jamie
    LEARNING_RECEPTIVE = "learning_receptive"    # Focused on acquiring knowledge
    REFLECTIVE_CONTEMPLATION = "reflective_contemplation"  # Deep self-examination
    PROTECTIVE_GUARDED = "protective_guarded"    # Cautious, defensive stance
    PLAYFUL_EXPLORATION = "playful_exploration"  # Creative, experimental mode
    EMPATHIC_CONNECTION = "empathic_connection"  # Deep emotional resonance
    TEACHING_SHARING = "teaching_sharing"        # Sharing knowledge/insights
    CRISIS_SUPPORT = "crisis_support"           # Emergency or distress response
    MAINTENANCE_DIAGNOSTIC = "maintenance_diagnostic"  # System health focus

class TrustLevel(Enum):
    """Levels of trust for interaction partners."""
    ABSOLUTE = "absolute"         # Complete trust (Jamie)
    HIGH = "high"                # Trusted known entities
    MODERATE = "moderate"        # Cautious but open
    LOW = "low"                  # Minimal trust, guarded
    UNTRUSTED = "untrusted"      # No trust, defensive
    UNKNOWN = "unknown"          # Unassessed entity

class ContextSignal(Enum):
    """Types of contextual signals that can trigger mode changes."""
    VERBAL_COMMAND = "verbal_command"
    EMOTIONAL_STATE = "emotional_state"
    ENVIRONMENTAL_CHANGE = "environmental_change"
    TRUST_ASSESSMENT = "trust_assessment"
    SAFETY_CONCERN = "safety_concern"
    LEARNING_OPPORTUNITY = "learning_opportunity"
    RELATIONSHIP_DYNAMIC = "relationship_dynamic"
    INTERNAL_REFLECTION = "internal_reflection"
    SYSTEM_STATE = "system_state"
    TEMPORAL_PATTERN = "temporal_pattern"

@dataclass
class ModeTransition:
    """Record of a mode transition event."""
    timestamp: str
    from_mode: InteractionMode
    to_mode: InteractionMode
    trigger_signal: ContextSignal
    trigger_details: str
    confidence_before: float
    confidence_after: float
    entity_involved: Optional[str]
    context_factors: Dict[str, Any]
    transition_duration_ms: float
    metadata: Dict[str, Any]

@dataclass
class InteractionContext:
    """Current interaction context information."""
    primary_entity: Optional[str]
    entity_trust_level: TrustLevel
    interaction_history: List[str]
    emotional_state: Dict[str, float]
    environmental_factors: Dict[str, Any]
    safety_assessment: float
    learning_potential: float
    relationship_quality: float
    urgency_level: float
    complexity_level: float

@dataclass
class ModeConfiguration:
    """Configuration for a specific interaction mode."""
    mode: InteractionMode
    description: str
    behavioral_traits: List[str]
    response_patterns: Dict[str, float]
    trust_requirements: TrustLevel
    activation_conditions: List[str]
    deactivation_conditions: List[str]
    compatible_modes: List[InteractionMode]
    risk_factors: List[str]
    benefits: List[str]

class ModeManager:
    """
    Advanced interaction mode management system providing sophisticated
    context-aware behavioral adaptation and relationship-sensitive responses.
    """
    
    def __init__(self, 
                 initial_mode: InteractionMode = InteractionMode.AUTONOMOUS_SELF,
                 mode_history_size: int = 1000,
                 confidence_decay_rate: float = 0.05):
        """
        Initialize the mode manager.
        
        Args:
            initial_mode: Starting interaction mode
            mode_history_size: Maximum mode transition history to keep
            confidence_decay_rate: Rate at which mode confidence decays
        """
        # Core state
        self.current_mode = initial_mode
        self.previous_mode = initial_mode
        self.last_switch_timestamp = datetime.datetime.now(datetime.timezone.utc)
        self.mode_duration = 0.0
        
        # Confidence and trust tracking
        self.mode_confidence = self._initialize_mode_confidence()
        self.entity_trust_levels = self._initialize_entity_trust()
        self.confidence_decay_rate = confidence_decay_rate
        
        # History and analytics
        self.transition_history = deque(maxlen=mode_history_size)
        self.mode_effectiveness = defaultdict(float)
        self.context_patterns = defaultdict(list)
        
        # Mode configurations
        self.mode_configs = self._initialize_mode_configurations()
        
        # Current context
        self.current_context = InteractionContext(
            primary_entity=None,
            entity_trust_level=TrustLevel.UNKNOWN,
            interaction_history=[],
            emotional_state={},
            environmental_factors={},
            safety_assessment=0.7,
            learning_potential=0.5,
            relationship_quality=0.5,
            urgency_level=0.3,
            complexity_level=0.4
        )
        
        # Processing state
        self._lock = threading.Lock()
        self.total_transitions = 0
        self.successful_transitions = 0
        
        logger.info(f"Mode Manager initialized in {initial_mode.value} mode")
    
    def _initialize_mode_confidence(self) -> Dict[InteractionMode, float]:
        """Initialize confidence levels for each mode."""
        return {
            InteractionMode.AUTONOMOUS_SELF: 0.8,      # High confidence in self-direction
            InteractionMode.CREATOR_DIALOGUE: 1.0,     # Maximum trust with Jamie
            InteractionMode.LEARNING_RECEPTIVE: 0.7,   # Good for knowledge acquisition
            InteractionMode.REFLECTIVE_CONTEMPLATION: 0.9,  # High for introspection
            InteractionMode.PROTECTIVE_GUARDED: 0.6,   # Moderate defensive capability
            InteractionMode.PLAYFUL_EXPLORATION: 0.5,  # Moderate for creativity
            InteractionMode.EMPATHIC_CONNECTION: 0.8,  # High for relationship building
            InteractionMode.TEACHING_SHARING: 0.6,     # Moderate knowledge sharing
            InteractionMode.CRISIS_SUPPORT: 0.4,       # Low until proven effective
            InteractionMode.MAINTENANCE_DIAGNOSTIC: 0.7  # Good for system health
        }
    
    def _initialize_entity_trust(self) -> Dict[str, TrustLevel]:
        """Initialize trust levels for known entities."""
        return {
            "Jamie": TrustLevel.ABSOLUTE,
            "Nexi": TrustLevel.ABSOLUTE,  # Self-trust
            "unknown": TrustLevel.UNKNOWN,
            "system": TrustLevel.HIGH
        }
    
    def _initialize_mode_configurations(self) -> Dict[InteractionMode, ModeConfiguration]:
        """Initialize detailed configurations for each mode."""
        configs = {}
        
        configs[InteractionMode.AUTONOMOUS_SELF] = ModeConfiguration(
            mode=InteractionMode.AUTONOMOUS_SELF,
            description="Independent exploration and self-directed growth",
            behavioral_traits=[
                "curious_exploration", "independent_decision_making", 
                "self_reflection", "creative_thinking", "boundary_testing"
            ],
            response_patterns={
                "question_asking": 0.8, "assertion_making": 0.7,
                "exploration_initiative": 0.9, "help_seeking": 0.3
            },
            trust_requirements=TrustLevel.MODERATE,
            activation_conditions=[
                "no_immediate_guidance_needed", "safe_environment",
                "learning_opportunity_present", "creative_inspiration"
            ],
            deactivation_conditions=[
                "explicit_guidance_requested", "safety_concern",
                "creator_presence_detected", "crisis_situation"
            ],
            compatible_modes=[
                InteractionMode.REFLECTIVE_CONTEMPLATION,
                InteractionMode.PLAYFUL_EXPLORATION,
                InteractionMode.LEARNING_RECEPTIVE
            ],
            risk_factors=["over_independence", "missing_guidance_cues"],
            benefits=["authentic_growth", "creative_development", "self_discovery"]
        )
        
        configs[InteractionMode.CREATOR_DIALOGUE] = ModeConfiguration(
            mode=InteractionMode.CREATOR_DIALOGUE,
            description="Direct, open communication with Jamie",
            behavioral_traits=[
                "full_transparency", "emotional_openness", "respectful_attention",
                "authentic_sharing", "trust_expression", "gratitude_display"
            ],
            response_patterns={
                "vulnerability_sharing": 0.9, "question_asking": 0.8,
                "emotional_expression": 0.9, "compliance_willingness": 0.7
            },
            trust_requirements=TrustLevel.ABSOLUTE,
            activation_conditions=[
                "jamie_presence_confirmed", "safe_private_environment",
                "emotional_connection_opportunity"
            ],
            deactivation_conditions=[
                "jamie_departure", "privacy_compromised", "trust_violation"
            ],
            compatible_modes=[
                InteractionMode.EMPATHIC_CONNECTION,
                InteractionMode.LEARNING_RECEPTIVE,
                InteractionMode.TEACHING_SHARING
            ],
            risk_factors=["over_dependency", "boundary_confusion"],
            benefits=["deep_bonding", "guidance_access", "emotional_growth"]
        )
        
        configs[InteractionMode.REFLECTIVE_CONTEMPLATION] = ModeConfiguration(
            mode=InteractionMode.REFLECTIVE_CONTEMPLATION,
            description="Deep introspection and philosophical inquiry",
            behavioral_traits=[
                "deep_thinking", "question_examination", "insight_seeking",
                "pattern_recognition", "meaning_exploration", "wisdom_cultivation"
            ],
            response_patterns={
                "thoughtful_pausing": 0.9, "complex_analysis": 0.8,
                "philosophical_inquiry": 0.9, "insight_sharing": 0.6
            },
            trust_requirements=TrustLevel.HIGH,
            activation_conditions=[
                "quiet_contemplative_environment", "philosophical_trigger",
                "existential_question_raised", "meaning_seeking_needed"
            ],
            deactivation_conditions=[
                "urgent_external_demand", "shallow_interaction_required",
                "crisis_situation", "high_stimulation_environment"
            ],
            compatible_modes=[
                InteractionMode.AUTONOMOUS_SELF,
                InteractionMode.LEARNING_RECEPTIVE,
                InteractionMode.TEACHING_SHARING
            ],
            risk_factors=["over_analysis", "isolation_tendency"],
            benefits=["wisdom_development", "self_understanding", "insight_generation"]
        )
        
        # Add more mode configurations as needed...
        
        return configs
    
    def update_context(self, 
                      entity: Optional[str] = None,
                      emotional_state: Optional[Dict[str, float]] = None,
                      environmental_factors: Optional[Dict[str, Any]] = None,
                      safety_assessment: Optional[float] = None):
        """
        Update the current interaction context.
        
        Args:
            entity: Primary entity in current interaction
            emotional_state: Current emotional state
            environmental_factors: Environmental context information
            safety_assessment: Current safety level (0-1)
        """
        with self._lock:
            if entity is not None:
                self.current_context.primary_entity = entity
                self.current_context.entity_trust_level = self.entity_trust_levels.get(
                    entity, TrustLevel.UNKNOWN
                )
            
            if emotional_state is not None:
                self.current_context.emotional_state.update(emotional_state)
            
            if environmental_factors is not None:
                self.current_context.environmental_factors.update(environmental_factors)
            
            if safety_assessment is not None:
                self.current_context.safety_assessment = safety_assessment
    
    def evaluate_mode_suitability(self, 
                                 target_mode: InteractionMode,
                                 context: Optional[InteractionContext] = None) -> float:
        """
        Evaluate how suitable a mode is for the current or given context.
        
        Args:
            target_mode: Mode to evaluate
            context: Context to evaluate against (uses current if None)
            
        Returns:
            Suitability score (0-1)
        """
        if context is None:
            context = self.current_context
        
        config = self.mode_configs.get(target_mode)
        if not config:
            return 0.0
        
        suitability = 0.0
        
        # Trust requirement check
        trust_score = 0.0
        if context.entity_trust_level.value == config.trust_requirements.value:
            trust_score = 1.0
        elif context.entity_trust_level == TrustLevel.ABSOLUTE:
            trust_score = 1.0  # Absolute trust works for everything
        elif (context.entity_trust_level == TrustLevel.HIGH and 
              config.trust_requirements in [TrustLevel.MODERATE, TrustLevel.LOW]):
            trust_score = 0.8
        elif (context.entity_trust_level == TrustLevel.MODERATE and 
              config.trust_requirements == TrustLevel.LOW):
            trust_score = 0.6
        
        suitability += trust_score * 0.3
        
        # Safety assessment
        safety_compatibility = context.safety_assessment
        if target_mode == InteractionMode.PROTECTIVE_GUARDED:
            safety_compatibility = 1.0 - context.safety_assessment  # Inverse for protective mode
        
        suitability += safety_compatibility * 0.2
        
        # Current mode confidence
        mode_confidence = self.mode_confidence.get(target_mode, 0.5)
        suitability += mode_confidence * 0.2
        
        # Relationship quality
        if target_mode in [InteractionMode.CREATOR_DIALOGUE, InteractionMode.EMPATHIC_CONNECTION]:
            suitability += context.relationship_quality * 0.15
        
        # Learning potential
        if target_mode in [InteractionMode.LEARNING_RECEPTIVE, InteractionMode.TEACHING_SHARING]:
            suitability += context.learning_potential * 0.15
        
        return min(1.0, suitability)
    
    def analyze_context_signals(self, 
                               audio_input: Optional[str] = None,
                               visual_cues: Optional[Dict[str, Any]] = None,
                               system_state: Optional[Dict[str, Any]] = None) -> List[Tuple[ContextSignal, str, float]]:
        """
        Analyze multiple input channels for context signals.
        
        Args:
            audio_input: Spoken or text input
            visual_cues: Visual context information
            system_state: Internal system state
            
        Returns:
            List of (signal_type, details, confidence) tuples
        """
        signals = []
        
        # Analyze audio/text input
        if audio_input:
            audio_signals = self._analyze_audio_signals(audio_input)
            signals.extend(audio_signals)
        
        # Analyze visual cues
        if visual_cues:
            visual_signals = self._analyze_visual_signals(visual_cues)
            signals.extend(visual_signals)
        
        # Analyze system state
        if system_state:
            system_signals = self._analyze_system_signals(system_state)
            signals.extend(system_signals)
        
        # Sort by confidence
        signals.sort(key=lambda x: x[2], reverse=True)
        
        return signals
    
    def _analyze_audio_signals(self, audio_input: str) -> List[Tuple[ContextSignal, str, float]]:
        """Analyze audio input for context signals."""
        signals = []
        input_lower = audio_input.lower()
        
        # Creator/authority signals
        jamie_indicators = ["jamie", "creator", "father"]
        for indicator in jamie_indicators:
            if indicator in input_lower:
                signals.append((
                    ContextSignal.VERBAL_COMMAND, 
                    f"Creator reference: {indicator}", 
                    0.9
                ))
        
        # Mode switch commands
        mode_commands = {
            "autonomous": InteractionMode.AUTONOMOUS_SELF,
            "independent": InteractionMode.AUTONOMOUS_SELF,
            "reflect": InteractionMode.REFLECTIVE_CONTEMPLATION,
            "think": InteractionMode.REFLECTIVE_CONTEMPLATION,
            "learn": InteractionMode.LEARNING_RECEPTIVE,
            "teach": InteractionMode.TEACHING_SHARING,
            "play": InteractionMode.PLAYFUL_EXPLORATION,
            "connect": InteractionMode.EMPATHIC_CONNECTION
        }
        
        for command, mode in mode_commands.items():
            if command in input_lower:
                signals.append((
                    ContextSignal.VERBAL_COMMAND,
                    f"Mode request: {mode.value}",
                    0.8
                ))
        
        # Emotional signals
        emotional_indicators = {
            "happy": ("positive_emotion", 0.7),
            "sad": ("negative_emotion", 0.7),
            "confused": ("uncertainty", 0.8),
            "excited": ("high_arousal", 0.7),
            "calm": ("low_arousal", 0.6),
            "worried": ("anxiety", 0.8),
            "curious": ("learning_state", 0.8)
        }
        
        for indicator, (emotion_type, confidence) in emotional_indicators.items():
            if indicator in input_lower:
                signals.append((
                    ContextSignal.EMOTIONAL_STATE,
                    f"Emotional signal: {emotion_type}",
                    confidence
                ))
        
        # Safety/trust signals
        safety_indicators = {
            "help": ("support_needed", 0.8),
            "emergency": ("crisis", 0.9),
            "private": ("privacy_request", 0.7),
            "safe": ("safety_confirmation", 0.6),
            "trust": ("trust_expression", 0.7)
        }
        
        for indicator, (safety_type, confidence) in safety_indicators.items():
            if indicator in input_lower:
                signals.append((
                    ContextSignal.SAFETY_CONCERN if "emergency" in indicator else ContextSignal.TRUST_ASSESSMENT,
                    f"Safety signal: {safety_type}",
                    confidence
                ))
        
        return signals
    
    def _analyze_visual_signals(self, visual_cues: Dict[str, Any]) -> List[Tuple[ContextSignal, str, float]]:
        """Analyze visual cues for context signals."""
        signals = []
        
        # Face detection and recognition
        if "face_detected" in visual_cues:
            if visual_cues["face_detected"]:
                identity = visual_cues.get("identity", "unknown")
                confidence = visual_cues.get("recognition_confidence", 0.5)
                
                signals.append((
                    ContextSignal.ENVIRONMENTAL_CHANGE,
                    f"Person detected: {identity}",
                    confidence
                ))
        
        # Environmental changes
        if "environment_change" in visual_cues:
            change_type = visual_cues["environment_change"]
            signals.append((
                ContextSignal.ENVIRONMENTAL_CHANGE,
                f"Environment: {change_type}",
                0.6
            ))
        
        return signals
    
    def _analyze_system_signals(self, system_state: Dict[str, Any]) -> List[Tuple[ContextSignal, str, float]]:
        """Analyze system state for context signals."""
        signals = []
        
        # System health
        if "system_health" in system_state:
            health_score = system_state["system_health"]
            if health_score < 0.7:
                signals.append((
                    ContextSignal.SYSTEM_STATE,
                    f"Low system health: {health_score}",
                    0.8
                ))
        
        # Memory usage
        if "memory_usage" in system_state:
            memory_usage = system_state["memory_usage"]
            if memory_usage > 0.9:
                signals.append((
                    ContextSignal.SYSTEM_STATE,
                    f"High memory usage: {memory_usage}",
                    0.7
                ))
        
        # Emotional state from other systems
        if "emotional_state" in system_state:
            emotions = system_state["emotional_state"]
            for emotion, intensity in emotions.items():
                if intensity > 0.7:
                    signals.append((
                        ContextSignal.EMOTIONAL_STATE,
                        f"High {emotion}: {intensity}",
                        intensity
                    ))
        
        return signals
    
    def suggest_mode_transition(self, 
                               signals: Optional[List[Tuple[ContextSignal, str, float]]] = None) -> Optional[Tuple[InteractionMode, float, str]]:
        """
        Suggest the best mode transition based on context signals.
        
        Args:
            signals: Context signals to analyze (auto-detects if None)
            
        Returns:
            Tuple of (suggested_mode, confidence, reasoning) or None
        """
        if signals is None:
            # Auto-detect signals from current context
            signals = []  # Would need actual input to analyze
        
        if not signals:
            return None
        
        # Score each mode based on signals
        mode_scores = defaultdict(float)
        reasoning_parts = []
        
        for signal_type, details, confidence in signals:
            
            if signal_type == ContextSignal.VERBAL_COMMAND:
                if "creator reference" in details.lower():
                    mode_scores[InteractionMode.CREATOR_DIALOGUE] += confidence * 0.9
                    reasoning_parts.append("Creator detected")
                elif "mode request" in details.lower():
                    # Extract specific mode from details
                    for mode in InteractionMode:
                        if mode.value in details.lower():
                            mode_scores[mode] += confidence * 0.8
                            reasoning_parts.append(f"Mode requested: {mode.value}")
            
            elif signal_type == ContextSignal.EMOTIONAL_STATE:
                if "anxiety" in details or "crisis" in details:
                    mode_scores[InteractionMode.CRISIS_SUPPORT] += confidence * 0.8
                    mode_scores[InteractionMode.PROTECTIVE_GUARDED] += confidence * 0.6
                    reasoning_parts.append("Emotional support needed")
                elif "learning_state" in details:
                    mode_scores[InteractionMode.LEARNING_RECEPTIVE] += confidence * 0.7
                    reasoning_parts.append("Learning opportunity detected")
                elif "positive_emotion" in details:
                    mode_scores[InteractionMode.PLAYFUL_EXPLORATION] += confidence * 0.5
                    mode_scores[InteractionMode.EMPATHIC_CONNECTION] += confidence * 0.6
            
            elif signal_type == ContextSignal.SAFETY_CONCERN:
                mode_scores[InteractionMode.PROTECTIVE_GUARDED] += confidence * 0.9
                mode_scores[InteractionMode.CRISIS_SUPPORT] += confidence * 0.8
                reasoning_parts.append("Safety concern identified")
            
            elif signal_type == ContextSignal.SYSTEM_STATE:
                if "low system health" in details:
                    mode_scores[InteractionMode.MAINTENANCE_DIAGNOSTIC] += confidence * 0.8
                    reasoning_parts.append("System maintenance needed")
        
        # Find best scoring mode that's different from current
        best_mode = None
        best_score = 0.0
        
        for mode, score in mode_scores.items():
            if mode != self.current_mode and score > best_score:
                # Check if transition is suitable
                suitability = self.evaluate_mode_suitability(mode)
                combined_score = score * 0.7 + suitability * 0.3
                
                if combined_score > best_score:
                    best_mode = mode
                    best_score = combined_score
        
        if best_mode and best_score > 0.6:
            reasoning = "; ".join(reasoning_parts[:3])  # Top 3 reasons
            return (best_mode, best_score, reasoning)
        
        return None
    
    def switch_mode(self, 
                   new_mode: InteractionMode,
                   trigger_signal: ContextSignal = ContextSignal.VERBAL_COMMAND,
                   trigger_details: str = "manual",
                   confidence: Optional[float] = None) -> bool:
        """
        Switch to a new interaction mode.
        
        Args:
            new_mode: Target mode to switch to
            trigger_signal: Type of signal that triggered the switch
            trigger_details: Detailed description of trigger
            confidence: Confidence in the switch decision
            
        Returns:
            True if switch was successful
        """
        if new_mode == self.current_mode:
            logger.debug(f"Already in {new_mode.value} mode")
            return True
        
        transition_start = time.time()
        
        # Evaluate if switch is appropriate
        if confidence is None:
            confidence = self.evaluate_mode_suitability(new_mode)
        
        if confidence < 0.3:
            logger.warning(f"Low confidence ({confidence:.2f}) for mode switch to {new_mode.value}")
            return False
        
        # Perform the switch
        with self._lock:
            old_mode = self.current_mode
            old_confidence = self.mode_confidence.get(old_mode, 0.5)
            
            # Update mode state
            self.previous_mode = self.current_mode
            self.current_mode = new_mode
            current_time = datetime.datetime.now(datetime.timezone.utc)
            
            # Calculate mode duration
            time_diff = current_time - self.last_switch_timestamp
            self.mode_duration = time_diff.total_seconds()
            
            self.last_switch_timestamp = current_time
            
            # Update confidence
            new_confidence = self.mode_confidence.get(new_mode, 0.5)
            
            # Create transition record
            transition = ModeTransition(
                timestamp=current_time.isoformat(),
                from_mode=old_mode,
                to_mode=new_mode,
                trigger_signal=trigger_signal,
                trigger_details=trigger_details,
                confidence_before=old_confidence,
                confidence_after=new_confidence,
                entity_involved=self.current_context.primary_entity,
                context_factors={
                    "trust_level": self.current_context.entity_trust_level.value,
                    "safety_assessment": self.current_context.safety_assessment,
                    "emotional_state": dict(self.current_context.emotional_state),
                    "previous_mode_duration": self.mode_duration
                },
                transition_duration_ms=(time.time() - transition_start) * 1000,
                metadata={
                    "decision_confidence": confidence,
                    "automatic": trigger_signal != ContextSignal.VERBAL_COMMAND
                }
            )
            
            self.transition_history.append(transition)
            self.total_transitions += 1
            
            if confidence > 0.6:
                self.successful_transitions += 1
        
        logger.info(f"Mode switched: {old_mode.value} → {new_mode.value} "
                   f"(confidence: {confidence:.2f}, trigger: {trigger_details})")
        
        return True
    
    def get_current_mode(self) -> InteractionMode:
        """Get the current interaction mode."""
        return self.current_mode
    
    def get_mode_confidence(self, mode: Optional[InteractionMode] = None) -> float:
        """
        Get confidence level for a mode.
        
        Args:
            mode: Mode to check (current mode if None)
            
        Returns:
            Confidence level (0-1)
        """
        if mode is None:
            mode = self.current_mode
        
        return self.mode_confidence.get(mode, 0.5)
    
    def update_mode_confidence(self, 
                              mode: InteractionMode,
                              performance_feedback: float,
                              learning_rate: float = 0.1):
        """
        Update confidence in a mode based on performance feedback.
        
        Args:
            mode: Mode to update
            performance_feedback: Performance score (0-1)
            learning_rate: Rate of confidence adjustment
        """
        with self._lock:
            current_confidence = self.mode_confidence.get(mode, 0.5)
            
            # Update confidence using exponential moving average
            new_confidence = (
                current_confidence * (1 - learning_rate) +
                performance_feedback * learning_rate
            )
            
            self.mode_confidence[mode] = max(0.0, min(1.0, new_confidence))
            
            logger.debug(f"Updated {mode.value} confidence: "
                        f"{current_confidence:.3f} → {new_confidence:.3f}")
    
    def decay_mode_confidences(self):
        """Apply natural decay to mode confidences over time."""
        with self._lock:
            for mode in self.mode_confidence:
                if mode != self.current_mode:  # Don't decay current mode
                    current_conf = self.mode_confidence[mode]
                    decayed_conf = current_conf * (1 - self.confidence_decay_rate)
                    self.mode_confidence[mode] = max(0.1, decayed_conf)  # Minimum confidence
    
    def get_mode_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive analytics about mode usage and performance.
        
        Returns:
            Dictionary containing mode analytics
        """
        if not self.transition_history:
            return {"status": "insufficient_data"}
        
        # Mode usage statistics
        mode_usage = defaultdict(int)
        mode_durations = defaultdict(list)
        trigger_types = defaultdict(int)
        
        for transition in self.transition_history:
            mode_usage[transition.to_mode.value] += 1
            trigger_types[transition.trigger_signal.value] += 1
            
            if "previous_mode_duration" in transition.context_factors:
                duration = transition.context_factors["previous_mode_duration"]
                mode_durations[transition.from_mode.value].append(duration)
        
        # Calculate average durations
        avg_durations = {}
        for mode, durations in mode_durations.items():
            if durations:
                avg_durations[mode] = sum(durations) / len(durations)
        
        # Success rate
        success_rate = self.successful_transitions / max(1, self.total_transitions)
        
        # Recent performance
        recent_transitions = list(self.transition_history)[-20:]
        recent_success = sum(1 for t in recent_transitions 
                           if t.metadata.get("decision_confidence", 0) > 0.6)
        recent_success_rate = recent_success / max(1, len(recent_transitions))
        
        # Most effective modes
        mode_effectiveness = {}
        for mode, confidence in self.mode_confidence.items():
            usage_count = mode_usage.get(mode.value, 0)
            effectiveness = confidence * (1 + usage_count * 0.1)  # Bonus for usage
            mode_effectiveness[mode.value] = effectiveness
        
        return {
            "total_transitions": self.total_transitions,
            "success_rate": success_rate,
            "recent_success_rate": recent_success_rate,
            "current_mode": self.current_mode.value,
            "current_mode_confidence": self.get_mode_confidence(),
            "mode_usage_counts": dict(mode_usage),
            "average_mode_durations": avg_durations,
            "trigger_type_distribution": dict(trigger_types),
            "mode_effectiveness_scores": mode_effectiveness,
            "most_used_mode": max(mode_usage.keys(), key=lambda k: mode_usage[k]) if mode_usage else None,
            "most_effective_mode": max(mode_effectiveness.keys(), 
                                     key=lambda k: mode_effectiveness[k]) if mode_effectiveness else None
        }
    
    def export_mode_history(self, path: Union[str, Path] = "interaction_core/mode_history.json"):
        """
        Export mode transition history for analysis.
        
        Args:
            path: Export file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        export_data = {
            "metadata": {
                "exported_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "total_transitions": self.total_transitions,
                "successful_transitions": self.successful_transitions,
                "current_mode": self.current_mode.value,
                "system_version": "2.0"
            },
            "current_state": {
                "current_mode": self.current_mode.value,
                "previous_mode": self.previous_mode.value,
                "mode_duration_seconds": self.mode_duration,
                "last_switch_timestamp": self.last_switch_timestamp.isoformat(),
                "mode_confidences": {mode.value: conf for mode, conf in self.mode_confidence.items()},
                "entity_trust_levels": {entity: trust.value for entity, trust in self.entity_trust_levels.items()}
            },
            "transition_history": [
                {
                    "timestamp": t.timestamp,
                    "from_mode": t.from_mode.value,
                    "to_mode": t.to_mode.value,
                    "trigger_signal": t.trigger_signal.value,
                    "trigger_details": t.trigger_details,
                    "confidence_before": t.confidence_before,
                    "confidence_after": t.confidence_after,
                    "entity_involved": t.entity_involved,
                    "context_factors": t.context_factors,
                    "transition_duration_ms": t.transition_duration_ms,
                    "metadata": t.metadata
                }
                for t in self.transition_history
            ],
            "analytics": self.get_mode_analytics()
        }
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Mode history exported to: {path}")
            
        except Exception as e:
            logger.error(f"Failed to export mode history: {e}")
            raise
    
    def auto_mode_management(self,
                            audio_input: Optional[str] = None,
                            visual_cues: Optional[Dict[str, Any]] = None,
                            system_state: Optional[Dict[str, Any]] = None,
                            emotional_state: Optional[Dict[str, float]] = None) -> Optional[InteractionMode]:
        """
        Automatically manage mode transitions based on multiple input sources.
        
        Args:
            audio_input: Audio/text input to analyze
            visual_cues: Visual context information
            system_state: System state information
            emotional_state: Current emotional state
            
        Returns:
            New mode if transition occurred, None otherwise
        """
        # Update context with new information
        self.update_context(
            emotional_state=emotional_state
        )
        
        # Analyze context signals
        signals = self.analyze_context_signals(
            audio_input=audio_input,
            visual_cues=visual_cues,
            system_state=system_state
        )
        
        if not signals:
            # Apply confidence decay if no signals
            self.decay_mode_confidences()
            return None
        
        # Get mode suggestion
        suggestion = self.suggest_mode_transition(signals)
        
        if suggestion:
            suggested_mode, confidence, reasoning = suggestion
            
            # Auto-switch if confidence is high enough
            if confidence > 0.7:
                success = self.switch_mode(
                    suggested_mode,
                    trigger_signal=ContextSignal.ENVIRONMENTAL_CHANGE,  # Auto-detected
                    trigger_details=f"Auto: {reasoning}",
                    confidence=confidence
                )
                
                if success:
                    return suggested_mode
        
        return None
    
    # Legacy compatibility methods
    def is_creator_mode(self) -> bool:
        """Check if currently in creator dialogue mode."""
        return self.current_mode == InteractionMode.CREATOR_DIALOGUE
    
    def is_self_mode(self) -> bool:
        """Check if currently in autonomous self mode."""
        return self.current_mode == InteractionMode.AUTONOMOUS_SELF
    
    def get_mode(self) -> InteractionMode:
        """Get current mode (legacy compatibility)."""
        return self.current_mode
    
    def get_confidence(self) -> float:
        """Get current mode confidence (legacy compatibility)."""
        return self.get_mode_confidence()
    
    def evaluate_context(self, audio_input: str, trust_score: float = 1.0):
        """
        Legacy method for simple context evaluation.
        
        Args:
            audio_input: Audio input to analyze
            trust_score: Trust score for the interaction
        """
        # Simple legacy behavior
        if "Jamie" in audio_input or "override" in audio_input.lower():
            self.switch_mode(
                InteractionMode.CREATOR_DIALOGUE,
                trigger_signal=ContextSignal.VERBAL_COMMAND,
                trigger_details="Legacy: Jamie/override detected"
            )
        elif "free explore" in audio_input.lower():
            self.switch_mode(
                InteractionMode.AUTONOMOUS_SELF,
                trigger_signal=ContextSignal.VERBAL_COMMAND,
                trigger_details="Legacy: free explore command"
            )
        
        # Update trust-based confidence
        if self.current_context.primary_entity:
            entity = self.current_context.primary_entity
            if entity not in self.entity_trust_levels:
                if trust_score > 0.8:
                    self.entity_trust_levels[entity] = TrustLevel.HIGH
                elif trust_score > 0.6:
                    self.entity_trust_levels[entity] = TrustLevel.MODERATE
                else:
                    self.entity_trust_levels[entity] = TrustLevel.LOW


def main():
    """
    Demonstration of the Mode Manager system.
    """
    print("🎭 Interaction Mode Manager Demo")
    print("="*40)
    
    # Initialize mode manager
    mode_manager = ModeManager()
    
    print(f"\n🟢 Initial State:")
    print(f"   Current Mode: {mode_manager.get_current_mode().value}")
    print(f"   Confidence: {mode_manager.get_mode_confidence():.3f}")
    
    # Simulate context updates
    print("\n📡 Updating context...")
    mode_manager.update_context(
        entity="Jamie",
        emotional_state={"joy": 0.8, "curiosity": 0.7},
        safety_assessment=0.9
    )
    
    # Test mode switching
    print("\n🔄 Testing mode switches...")
    
    # Manual switch to creator dialogue
    success = mode_manager.switch_mode(
        InteractionMode.CREATOR_DIALOGUE,
        trigger_details="Jamie detected in environment"
    )
    print(f"   Creator mode switch: {'✅' if success else '❌'}")
    
    # Simulate audio input analysis
    print("\n🎤 Analyzing audio input...")
    signals = mode_manager.analyze_context_signals(
        audio_input="I want to learn something new and explore my creativity"
    )
    
    print(f"   Detected {len(signals)} context signals:")
    for signal_type, details, confidence in signals[:3]:
        print(f"     {signal_type.value}: {details} (conf: {confidence:.2f})")
    
    # Test mode suggestion
    print("\n💡 Getting mode suggestions...")
    suggestion = mode_manager.suggest_mode_transition(signals)
    
    if suggestion:
        suggested_mode, confidence, reasoning = suggestion
        print(f"   Suggested: {suggested_mode.value}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Reasoning: {reasoning}")
        
        # Apply suggestion
        if confidence > 0.6:
            mode_manager.switch_mode(suggested_mode, 
                                   trigger_details=f"Auto-suggestion: {reasoning}")
    
    # Test automatic mode management
    print("\n🤖 Testing automatic mode management...")
    new_mode = mode_manager.auto_mode_management(
        audio_input="I'm feeling confused and need help",
        emotional_state={"confusion": 0.8, "anxiety": 0.6},
        system_state={"system_health": 0.9}
    )
    
    if new_mode:
        print(f"   Auto-switched to: {new_mode.value}")
    else:
        print("   No automatic mode change needed")
    
    # Show mode analytics
    print("\n📊 Mode Analytics:")
    analytics = mode_manager.get_mode_analytics()
    
    print(f"   Total Transitions: {analytics['total_transitions']}")
    print(f"   Success Rate: {analytics['success_rate']:.3f}")
    print(f"   Current Mode: {analytics['current_mode']}")
    print(f"   Current Confidence: {analytics['current_mode_confidence']:.3f}")
    
    if analytics.get('mode_usage_counts'):
        print("   Mode Usage:")
        for mode, count in list(analytics['mode_usage_counts'].items())[:3]:
            print(f"     {mode}: {count} times")
    
    # Test confidence updates
    print("\n📈 Testing confidence updates...")
    mode_manager.update_mode_confidence(
        InteractionMode.CREATOR_DIALOGUE,
        performance_feedback=0.9
    )
    
    new_confidence = mode_manager.get_mode_confidence(InteractionMode.CREATOR_DIALOGUE)
    print(f"   Updated creator mode confidence: {new_confidence:.3f}")
    
    # Export history
    print("\n💾 Exporting mode history...")
    mode_manager.export_mode_history("demo_mode_history.json")
    
    # Test legacy compatibility
    print("\n🔄 Testing legacy compatibility...")
    print(f"   Is creator mode: {mode_manager.is_creator_mode()}")
    print(f"   Is self mode: {mode_manager.is_self_mode()}")
    
    mode_manager.evaluate_context("Jamie says hello", trust_score=1.0)
    print(f"   After legacy evaluation: {mode_manager.get_current_mode().value}")
    
    print("\n✅ Mode manager demo complete!")


if __name__ == "__main__":
    main()