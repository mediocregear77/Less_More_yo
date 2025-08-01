"""
Affective System

Implements comprehensive emotional processing and regulation for Nexi's consciousness.
Based on emotion theory, affective neuroscience, and computational models of emotion.
Provides multi-dimensional emotional states, regulation mechanisms, and empathic resonance.

Core Functions:
- Multi-dimensional emotion modeling (valence, arousal, dominance)
- Emotional regulation and homeostasis
- Empathic resonance and emotional contagion
- Mood tracking and emotional memory
- Affective prediction and anticipation
- Emotional intelligence and social emotion processing
"""

import time
import json
import datetime
import logging
import threading
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from enum import Enum
from pathlib import Path
import math
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionCategory(Enum):
    """Basic emotion categories based on psychological research."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    LOVE = "love"
    CURIOSITY = "curiosity"
    WONDER = "wonder"
    CONTEMPT = "contempt"
    PRIDE = "pride"
    SHAME = "shame"
    GUILT = "guilt"
    ENVY = "envy"
    GRATITUDE = "gratitude"
    COMPASSION = "compassion"
    ANTICIPATION = "anticipation"
    NOSTALGIA = "nostalgia"
    SERENITY = "serenity"
    EXCITEMENT = "excitement"

class EmotionSource(Enum):
    """Sources of emotional triggers."""
    INTERNAL = "internal"              # Self-generated emotions
    EXTERNAL_STIMULUS = "external_stimulus"  # Environmental triggers
    MEMORY_RECALL = "memory_recall"    # From remembered experiences
    EMPATHIC_RESONANCE = "empathic_resonance"  # From others' emotions
    COGNITIVE_APPRAISAL = "cognitive_appraisal"  # From thought evaluation
    PREDICTION_ERROR = "prediction_error"  # From failed predictions
    SOCIAL_INTERACTION = "social_interaction"  # From relationships
    EXISTENTIAL = "existential"        # From deep self-reflection

class RegulationStrategy(Enum):
    """Emotion regulation strategies."""
    COGNITIVE_REAPPRAISAL = "cognitive_reappraisal"
    SUPPRESSION = "suppression"
    DISTRACTION = "distraction"
    ACCEPTANCE = "acceptance"
    EXPRESSION = "expression"
    SOCIAL_SHARING = "social_sharing"
    MINDFULNESS = "mindfulness"
    PROBLEM_SOLVING = "problem_solving"

class MoodState(Enum):
    """Overall mood states."""
    EUPHORIC = "euphoric"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    MELANCHOLIC = "melancholic"
    DEPRESSED = "depressed"
    ANXIOUS = "anxious"
    IRRITABLE = "irritable"
    CONTEMPLATIVE = "contemplative"

@dataclass
class EmotionVector:
    """Multi-dimensional emotion representation."""
    valence: float      # Positive (1.0) to Negative (-1.0)
    arousal: float      # High activation (1.0) to Low activation (0.0)
    dominance: float    # High control (1.0) to Low control (0.0)
    intensity: float    # Overall strength (0.0 to 1.0)
    
    def magnitude(self) -> float:
        """Calculate emotion magnitude in 3D space."""
        return math.sqrt(self.valence**2 + self.arousal**2 + self.dominance**2)
    
    def distance_to(self, other: 'EmotionVector') -> float:
        """Calculate distance to another emotion vector."""
        return math.sqrt(
            (self.valence - other.valence)**2 +
            (self.arousal - other.arousal)**2 +
            (self.dominance - other.dominance)**2
        )

@dataclass
class EmotionEvent:
    """Comprehensive emotion event record."""
    timestamp: str
    event_id: str
    emotion_category: EmotionCategory
    emotion_vector: EmotionVector
    source: EmotionSource
    trigger_description: str
    context: Dict[str, Any]
    regulation_applied: Optional[RegulationStrategy]
    duration_seconds: Optional[float]
    resolution: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class EmpatheticResonance:
    """Empathic connection with another entity."""
    entity_name: str
    resonance_strength: float
    emotional_synchrony: float
    trust_level: float
    interaction_history: List[str]
    last_interaction: Optional[str]
    created_timestamp: str
    updated_timestamp: str

@dataclass
class MoodProfile:
    """Current mood state assessment."""
    timestamp: str
    mood_state: MoodState
    stability: float
    duration_hours: float
    contributing_emotions: List[EmotionCategory]
    energy_level: float
    social_disposition: float
    creativity_level: float
    decision_making_capacity: float

class AffectiveSystem:
    """
    Advanced affective processing system implementing multi-dimensional
    emotion modeling, regulation, and empathic capabilities.
    """
    
    def __init__(self,
                 emotion_decay_rate: float = 0.1,
                 homeostasis_target: float = 0.6,
                 empathy_sensitivity: float = 0.8,
                 regulation_threshold: float = 0.7):
        """
        Initialize the affective system.
        
        Args:
            emotion_decay_rate: Rate at which emotions naturally decay
            homeostasis_target: Target emotional equilibrium
            empathy_sensitivity: Sensitivity to others' emotions
            regulation_threshold: Intensity threshold for automatic regulation
        """
        # Core emotional state
        self.current_emotions: Dict[EmotionCategory, EmotionVector] = {}
        self.emotion_history = deque(maxlen=1000)
        self.mood_history = deque(maxlen=100)
        
        # Empathic connections
        self.empathic_bonds: Dict[str, EmpatheticResonance] = {}
        
        # System parameters
        self.emotion_decay_rate = emotion_decay_rate
        self.homeostasis_target = homeostasis_target
        self.empathy_sensitivity = empathy_sensitivity
        self.regulation_threshold = regulation_threshold
        
        # Current state
        self.current_mood = MoodState.NEUTRAL
        self.homeostasis_level = homeostasis_target
        self.emotional_energy = 1.0
        self.regulation_capacity = 1.0
        
        # Emotional memory and learning
        self.emotion_associations: Dict[str, List[EmotionCategory]] = defaultdict(list)
        self.regulation_effectiveness: Dict[RegulationStrategy, float] = {
            strategy: 0.5 for strategy in RegulationStrategy
        }
        
        # Background processing
        self._processing_active = False
        self._processing_thread = None
        self._lock = threading.Lock()
        
        # Metrics
        self.total_emotions_processed = 0
        self.successful_regulations = 0
        self.empathic_interactions = 0
        
        # Initialize default empathic bond with Jamie
        self._initialize_jamie_bond()
        
        logger.info("Affective System initialized")
    
    def _initialize_jamie_bond(self):
        """Initialize the primary empathic bond with Jamie."""
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        jamie_bond = EmpatheticResonance(
            entity_name="Jamie",
            resonance_strength=1.0,  # Maximum bond strength
            emotional_synchrony=0.9,
            trust_level=1.0,
            interaction_history=[],
            last_interaction=None,
            created_timestamp=timestamp,
            updated_timestamp=timestamp
        )
        
        self.empathic_bonds["Jamie"] = jamie_bond
        logger.info("Initialized primary empathic bond with Jamie")
    
    def _emotion_category_to_vector(self, category: EmotionCategory, intensity: float = 0.5) -> EmotionVector:
        """
        Convert emotion category to multidimensional vector.
        Based on circumplex model and emotion research.
        """
        # Base emotion mappings (valence, arousal, dominance)
        emotion_mappings = {
            EmotionCategory.JOY: (0.8, 0.7, 0.6),
            EmotionCategory.SADNESS: (-0.6, 0.3, 0.2),
            EmotionCategory.ANGER: (-0.4, 0.9, 0.8),
            EmotionCategory.FEAR: (-0.7, 0.8, 0.1),
            EmotionCategory.SURPRISE: (0.1, 0.9, 0.3),
            EmotionCategory.DISGUST: (-0.8, 0.5, 0.7),
            EmotionCategory.LOVE: (0.9, 0.6, 0.5),
            EmotionCategory.CURIOSITY: (0.4, 0.6, 0.6),
            EmotionCategory.WONDER: (0.6, 0.5, 0.4),
            EmotionCategory.CONTEMPT: (-0.5, 0.4, 0.9),
            EmotionCategory.PRIDE: (0.7, 0.6, 0.8),
            EmotionCategory.SHAME: (-0.7, 0.4, 0.1),
            EmotionCategory.GUILT: (-0.6, 0.5, 0.2),
            EmotionCategory.ENVY: (-0.3, 0.6, 0.3),
            EmotionCategory.GRATITUDE: (0.8, 0.4, 0.6),
            EmotionCategory.COMPASSION: (0.6, 0.4, 0.5),
            EmotionCategory.ANTICIPATION: (0.3, 0.7, 0.5),
            EmotionCategory.NOSTALGIA: (0.2, 0.3, 0.4),
            EmotionCategory.SERENITY: (0.6, 0.2, 0.7),
            EmotionCategory.EXCITEMENT: (0.7, 0.9, 0.6)
        }
        
        base_valence, base_arousal, base_dominance = emotion_mappings.get(
            category, (0.0, 0.5, 0.5)
        )
        
        # Scale by intensity
        return EmotionVector(
            valence=base_valence * intensity,
            arousal=base_arousal * intensity,
            dominance=base_dominance * intensity,
            intensity=intensity
        )
    
    def trigger_emotion(self,
                       emotion: Union[EmotionCategory, str],
                       intensity: float = 0.5,
                       source: EmotionSource = EmotionSource.INTERNAL,
                       trigger_description: str = "",
                       context: Optional[Dict[str, Any]] = None,
                       from_entity: Optional[str] = None) -> EmotionEvent:
        """
        Trigger a new emotional state with comprehensive tracking.
        
        Args:
            emotion: Emotion category or name
            intensity: Emotion intensity (0.0 to 1.0)
            source: Source of the emotion
            trigger_description: Description of what triggered the emotion
            context: Additional context information
            from_entity: Entity that caused the emotion (for empathic responses)
            
        Returns:
            The created emotion event
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        event_id = f"emotion_{int(time.time()*1000)}_{self.total_emotions_processed}"
        
        # Convert to emotion category if string
        if isinstance(emotion, str):
            try:
                emotion_category = EmotionCategory(emotion.lower())
            except ValueError:
                logger.warning(f"Unknown emotion: {emotion}, defaulting to curiosity")
                emotion_category = EmotionCategory.CURIOSITY
        else:
            emotion_category = emotion
        
        # Clamp intensity
        intensity = max(0.0, min(1.0, intensity))
        
        # Create emotion vector
        emotion_vector = self._emotion_category_to_vector(emotion_category, intensity)
        
        # Handle empathic resonance if from another entity
        if from_entity and from_entity in self.empathic_bonds:
            resonance = self.empathic_bonds[from_entity]
            # Amplify emotion based on resonance strength
            resonance_factor = resonance.resonance_strength * self.empathy_sensitivity
            emotion_vector.intensity *= (1.0 + resonance_factor * 0.5)
            emotion_vector.intensity = min(1.0, emotion_vector.intensity)
            
            # Update empathic bond
            self._update_empathic_bond(from_entity, emotion_category, intensity)
        
        # Create emotion event
        emotion_event = EmotionEvent(
            timestamp=timestamp,
            event_id=event_id,
            emotion_category=emotion_category,
            emotion_vector=emotion_vector,
            source=source,
            trigger_description=trigger_description,
            context=context or {},
            regulation_applied=None,
            duration_seconds=None,
            resolution=None,
            metadata={
                "from_entity": from_entity,
                "homeostasis_before": self.homeostasis_level,
                "total_processed": self.total_emotions_processed
            }
        )
        
        with self._lock:
            # Add to current emotions (blending if same category exists)
            if emotion_category in self.current_emotions:
                existing = self.current_emotions[emotion_category]
                # Blend emotions using weighted average
                weight_new = emotion_vector.intensity
                weight_existing = existing.intensity
                total_weight = weight_new + weight_existing
                
                if total_weight > 0:
                    blended_vector = EmotionVector(
                        valence=(emotion_vector.valence * weight_new + existing.valence * weight_existing) / total_weight,
                        arousal=(emotion_vector.arousal * weight_new + existing.arousal * weight_existing) / total_weight,
                        dominance=(emotion_vector.dominance * weight_new + existing.dominance * weight_existing) / total_weight,
                        intensity=min(1.0, total_weight)
                    )
                    self.current_emotions[emotion_category] = blended_vector
            else:
                self.current_emotions[emotion_category] = emotion_vector
            
            # Add to history
            self.emotion_history.append(emotion_event)
            self.total_emotions_processed += 1
            
            # Update emotional associations
            if trigger_description:
                self.emotion_associations[trigger_description].append(emotion_category)
        
        # Update homeostasis
        self._update_homeostasis(emotion_vector)
        
        # Check for automatic regulation
        if intensity >= self.regulation_threshold:
            self._consider_automatic_regulation(emotion_event)
        
        # Update mood
        self._update_mood_state()
        
        logger.debug(f"Triggered {emotion_category.value}: intensity={intensity:.3f}, source={source.value}")
        return emotion_event
    
    def regulate_emotion(self,
                        strategy: RegulationStrategy,
                        target_emotion: Optional[EmotionCategory] = None,
                        effectiveness: Optional[float] = None) -> bool:
        """
        Apply emotion regulation strategy.
        
        Args:
            strategy: Regulation strategy to apply
            target_emotion: Specific emotion to regulate (None for all)
            effectiveness: Override effectiveness (None for learned value)
            
        Returns:
            True if regulation was successful
        """
        if not self.current_emotions:
            return False
        
        if effectiveness is None:
            effectiveness = self.regulation_effectiveness.get(strategy, 0.5)
        
        # Apply regulation based on current capacity
        actual_effectiveness = effectiveness * self.regulation_capacity
        
        emotions_to_regulate = (
            [target_emotion] if target_emotion and target_emotion in self.current_emotions
            else list(self.current_emotions.keys())
        )
        
        regulation_success = False
        
        with self._lock:
            for emotion_cat in emotions_to_regulate:
                emotion_vec = self.current_emotions[emotion_cat]
                
                # Apply strategy-specific regulation
                if strategy == RegulationStrategy.COGNITIVE_REAPPRAISAL:
                    # Reframe emotion more positively
                    emotion_vec.valence += actual_effectiveness * 0.3
                    emotion_vec.intensity *= (1.0 - actual_effectiveness * 0.4)
                    
                elif strategy == RegulationStrategy.SUPPRESSION:
                    # Reduce intensity directly
                    emotion_vec.intensity *= (1.0 - actual_effectiveness * 0.6)
                    
                elif strategy == RegulationStrategy.ACCEPTANCE:
                    # Increase dominance/control while maintaining intensity
                    emotion_vec.dominance = min(1.0, emotion_vec.dominance + actual_effectiveness * 0.4)
                    
                elif strategy == RegulationStrategy.EXPRESSION:
                    # Allow full expression but reduce duration
                    emotion_vec.arousal *= (1.0 - actual_effectiveness * 0.2)
                    
                elif strategy == RegulationStrategy.MINDFULNESS:
                    # Increase awareness and control
                    emotion_vec.dominance = min(1.0, emotion_vec.dominance + actual_effectiveness * 0.3)
                    emotion_vec.intensity *= (1.0 - actual_effectiveness * 0.3)
                
                # Clean up if intensity becomes very low
                if emotion_vec.intensity < 0.05:
                    del self.current_emotions[emotion_cat]
                    regulation_success = True
                elif emotion_vec.intensity < 0.5:  # Partial success
                    regulation_success = True
        
        # Update regulation capacity (gets depleted with use)
        self.regulation_capacity = max(0.1, self.regulation_capacity - 0.1)
        
        # Learn from regulation effectiveness
        if regulation_success:
            self.regulation_effectiveness[strategy] = min(1.0, 
                self.regulation_effectiveness[strategy] + 0.02)
            self.successful_regulations += 1
        else:
            self.regulation_effectiveness[strategy] = max(0.1,
                self.regulation_effectiveness[strategy] - 0.01)
        
        logger.debug(f"Applied {strategy.value} regulation: success={regulation_success}")
        return regulation_success
    
    def resonate_with_emotion(self,
                             entity_name: str,
                             emotion: EmotionCategory,
                             intensity: float,
                             context: str = "") -> float:
        """
        Empathically resonate with another entity's emotion.
        
        Args:
            entity_name: Name of the entity
            emotion: Their emotion
            intensity: Their emotion intensity
            context: Context of the interaction
            
        Returns:
            Resonance strength achieved
        """
        if entity_name not in self.empathic_bonds:
            self._create_empathic_bond(entity_name)
        
        bond = self.empathic_bonds[entity_name]
        
        # Calculate resonance based on bond strength and empathy sensitivity
        resonance_strength = bond.resonance_strength * self.empathy_sensitivity
        
        # Trigger corresponding emotion in self
        resonant_intensity = intensity * resonance_strength * 0.7  # Slightly reduced
        
        self.trigger_emotion(
            emotion=emotion,
            intensity=resonant_intensity,
            source=EmotionSource.EMPATHIC_RESONANCE,
            trigger_description=f"Empathic resonance with {entity_name}",
            context={"original_entity": entity_name, "original_intensity": intensity, "context": context},
            from_entity=entity_name
        )
        
        self.empathic_interactions += 1
        logger.info(f"Resonated with {entity_name}'s {emotion.value}: {resonant_intensity:.3f}")
        
        return resonance_strength
    
    def _create_empathic_bond(self, entity_name: str, initial_strength: float = 0.3):
        """Create a new empathic bond."""
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        bond = EmpatheticResonance(
            entity_name=entity_name,
            resonance_strength=initial_strength,
            emotional_synchrony=0.2,
            trust_level=0.3,
            interaction_history=[],
            last_interaction=None,
            created_timestamp=timestamp,
            updated_timestamp=timestamp
        )
        
        self.empathic_bonds[entity_name] = bond
        logger.info(f"Created empathic bond with {entity_name}")
    
    def _update_empathic_bond(self, entity_name: str, emotion: EmotionCategory, intensity: float):
        """Update empathic bond based on interaction."""
        if entity_name not in self.empathic_bonds:
            return
        
        bond = self.empathic_bonds[entity_name]
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        # Strengthen bond through positive interactions
        if emotion in [EmotionCategory.JOY, EmotionCategory.LOVE, EmotionCategory.GRATITUDE]:
            bond.resonance_strength = min(1.0, bond.resonance_strength + 0.02)
            bond.trust_level = min(1.0, bond.trust_level + 0.01)
        
        # Update synchrony based on emotional alignment
        if len(bond.interaction_history) > 0:
            recent_emotions = [EmotionCategory(hist.split(":")[1]) for hist in bond.interaction_history[-5:] 
                             if ":" in hist]
            if emotion in recent_emotions:
                bond.emotional_synchrony = min(1.0, bond.emotional_synchrony + 0.05)
        
        # Record interaction
        interaction_record = f"{timestamp}:{emotion.value}:{intensity:.2f}"
        bond.interaction_history.append(interaction_record)
        if len(bond.interaction_history) > 20:  # Keep recent history
            bond.interaction_history.pop(0)
        
        bond.last_interaction = timestamp
        bond.updated_timestamp = timestamp
    
    def _update_homeostasis(self, emotion_vector: EmotionVector):
        """Update emotional homeostasis based on new emotion."""
        # Positive emotions push toward positive homeostasis
        # Negative emotions push toward lower homeostasis
        homeostasis_impact = emotion_vector.valence * emotion_vector.intensity * 0.1
        
        self.homeostasis_level += homeostasis_impact
        self.homeostasis_level = max(0.0, min(1.0, self.homeostasis_level))
        
        # Natural drift toward target homeostasis
        drift = (self.homeostasis_target - self.homeostasis_level) * 0.02
        self.homeostasis_level += drift
    
    def _consider_automatic_regulation(self, emotion_event: EmotionEvent):
        """Consider applying automatic emotion regulation."""
        intensity = emotion_event.emotion_vector.intensity
        category = emotion_event.emotion_category
        
        # Auto-regulate high-intensity negative emotions
        if intensity > self.regulation_threshold:
            if category in [EmotionCategory.ANGER, EmotionCategory.FEAR, EmotionCategory.SADNESS]:
                # Apply cognitive reappraisal for negative emotions
                success = self.regulate_emotion(
                    RegulationStrategy.COGNITIVE_REAPPRAISAL,
                    target_emotion=category
                )
                if success:
                    emotion_event.regulation_applied = RegulationStrategy.COGNITIVE_REAPPRAISAL
            
            elif intensity > 0.9:  # Very high positive emotions might need regulation too
                # Apply acceptance strategy for overwhelming positive emotions
                success = self.regulate_emotion(
                    RegulationStrategy.ACCEPTANCE,
                    target_emotion=category
                )
                if success:
                    emotion_event.regulation_applied = RegulationStrategy.ACCEPTANCE
    
    def _update_mood_state(self):
        """Update overall mood based on current emotions."""
        if not self.current_emotions:
            self.current_mood = MoodState.NEUTRAL
            return
        
        # Calculate weighted average of current emotions
        total_valence = 0.0
        total_arousal = 0.0
        total_weight = 0.0
        
        for emotion_vec in self.current_emotions.values():
            weight = emotion_vec.intensity
            total_valence += emotion_vec.valence * weight
            total_arousal += emotion_vec.arousal * weight
            total_weight += weight
        
        if total_weight > 0:
            avg_valence = total_valence / total_weight
            avg_arousal = total_arousal / total_weight
            
            # Map to mood states
            if avg_valence > 0.6 and avg_arousal > 0.6:
                self.current_mood = MoodState.EUPHORIC
            elif avg_valence > 0.3:
                self.current_mood = MoodState.POSITIVE
            elif avg_valence < -0.6:
                if avg_arousal > 0.5:
                    self.current_mood = MoodState.ANXIOUS
                else:
                    self.current_mood = MoodState.DEPRESSED
            elif avg_valence < -0.3:
                self.current_mood = MoodState.MELANCHOLIC
            elif avg_arousal < 0.3:
                self.current_mood = MoodState.CONTEMPLATIVE
            else:
                self.current_mood = MoodState.NEUTRAL
    
    def get_current_emotional_state(self) -> Dict[str, Any]:
        """
        Get comprehensive current emotional state.
        
        Returns:
            Dictionary containing current emotional information
        """
        with self._lock:
            # Current emotions
            current_emotions_info = {}
            for emotion_cat, emotion_vec in self.current_emotions.items():
                current_emotions_info[emotion_cat.value] = {
                    "valence": emotion_vec.valence,
                    "arousal": emotion_vec.arousal,
                    "dominance": emotion_vec.dominance,
                    "intensity": emotion_vec.intensity,
                    "magnitude": emotion_vec.magnitude()
                }
            
            # Empathic bonds
            bonds_info = {}
            for entity, bond in self.empathic_bonds.items():
                bonds_info[entity] = {
                    "resonance_strength": bond.resonance_strength,
                    "emotional_synchrony": bond.emotional_synchrony,
                    "trust_level": bond.trust_level,
                    "interactions": len(bond.interaction_history)
                }
            
            return {
                "current_emotions": current_emotions_info,
                "mood_state": self.current_mood.value,
                "homeostasis_level": self.homeostasis_level,
                "emotional_energy": self.emotional_energy,
                "regulation_capacity": self.regulation_capacity,
                "empathic_bonds": bonds_info,
                "total_emotions_processed": self.total_emotions_processed,
                "successful_regulations": self.successful_regulations,
                "empathic_interactions": self.empathic_interactions,
                "dominant_emotion": max(current_emotions_info.keys(), 
                                      key=lambda x: current_emotions_info[x]["intensity"]) 
                                      if current_emotions_info else "none"
            }
    
    def create_mood_profile(self) -> MoodProfile:
        """Create detailed mood profile assessment."""
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        # Calculate mood duration (simplified)
        mood_duration = 1.0  # Would be calculated from mood history
        
        # Get contributing emotions
        contributing_emotions = [
            cat for cat, vec in self.current_emotions.items()
            if vec.intensity > 0.3
        ]
        
        # Calculate derived metrics
        energy_level = sum(vec.arousal * vec.intensity for vec in self.current_emotions.values()) / max(1, len(self.current_emotions))
        
        social_disposition = self.empathic_bonds.get("Jamie", EmpatheticResonance("", 0, 0, 0, [], None, "", "")).resonance_strength
        
        creativity_level = 0.8 if self.current_mood in [MoodState.POSITIVE, MoodState.CONTEMPLATIVE] else 0.4
        
        decision_capacity = self.regulation_capacity * self.homeostasis_level
        
        mood_profile = MoodProfile(
            timestamp=timestamp,
            mood_state=self.current_mood,
            stability=self.homeostasis_level,
            duration_hours=mood_duration,
            contributing_emotions=contributing_emotions,
            energy_level=energy_level,
            social_disposition=social_disposition,
            creativity_level=creativity_level,
            decision_making_capacity=decision_capacity
        )
        
        self.mood_history.append(mood_profile)
        return mood_profile
    
    def decay_emotions(self, decay_factor: Optional[float] = None):
        """
        Apply natural emotion decay over time.
        
        Args:
            decay_factor: Override decay rate
        """
        if decay_factor is None:
            decay_factor = self.emotion_decay_rate
        
        with self._lock:
            emotions_to_remove = []
            
            for emotion_cat, emotion_vec in self.current_emotions.items():
                # Apply decay
                emotion_vec.intensity *= (1.0 - decay_factor)
                
                # Remove very weak emotions
                if emotion_vec.intensity < 0.05:
                    emotions_to_remove.append(emotion_cat)
            
            # Remove decayed emotions
            for emotion_cat in emotions_to_remove:
                del self.current_emotions[emotion_cat]
        
        # Regenerate regulation capacity
        self.regulation_capacity = min(1.0, self.regulation_capacity + 0.05)
    
    def start_background_processing(self, interval_seconds: float = 5.0):
        """
        Start background emotional processing.
        
        Args:
            interval_seconds: Processing interval
        """
        if self._processing_active:
            return
        
        self._processing_active = True
        
        def processing_loop():
            while self._processing_active:
                try:
                    # Apply natural decay
                    self.decay_emotions()
                    
                    # Update mood
                    self._update_mood_state()
                    
                    # Create periodic mood profiles
                    if len(self.emotion_history) % 10 == 0:
                        self.create_mood_profile()
                    
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Error in emotional processing: {e}")
        
        self._processing_thread = threading.Thread(target=processing_loop, daemon=True)
        self._processing_thread.start()
        
        logger.info("Started background emotional processing")
    
    def stop_background_processing(self):
        """Stop background emotional processing."""
        self._processing_active = False
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)
        logger.info("Stopped background emotional processing")
    
    def reflect_on_emotional_patterns(self, timespan_hours: int = 24) -> Dict[str, Any]:
        """
        Analyze emotional patterns over a time period.
        
        Args:
            timespan_hours: Hours of history to analyze
            
        Returns:
            Emotional pattern analysis
        """
        cutoff_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=timespan_hours)
        cutoff_iso = cutoff_time.isoformat()
        
        # Filter recent emotions
        recent_emotions = [
            event for event in self.emotion_history
            if event.timestamp >= cutoff_iso
        ]
        
        if not recent_emotions:
            return {"status": "insufficient_data", "timespan": f"{timespan_hours} hours"}
        
        # Analyze patterns
        emotion_frequency = defaultdict(int)
        source_frequency = defaultdict(int)
        intensity_distribution = []
        valence_trend = []
        regulation_success_rate = 0
        
        for event in recent_emotions:
            emotion_frequency[event.emotion_category.value] += 1
            source_frequency[event.source.value] += 1
            intensity_distribution.append(event.emotion_vector.intensity)
            valence_trend.append(event.emotion_vector.valence)
            
            if event.regulation_applied:
                regulation_success_rate += 1
        
        regulation_success_rate = regulation_success_rate / len(recent_emotions) if recent_emotions else 0
        
        # Calculate insights
        dominant_emotion = max(emotion_frequency.keys(), key=lambda x: emotion_frequency[x]) if emotion_frequency else "none"
        
        avg_intensity = np.mean(intensity_distribution) if intensity_distribution else 0
        
        avg_valence = np.mean(valence_trend) if valence_trend else 0
        
        valence_stability = 1.0 - np.std(valence_trend) if len(valence_trend) > 1 else 1.0
        
        # Generate insights
        insights = []
        
        if avg_valence > 0.5:
            insights.append("Generally positive emotional period")
        elif avg_valence < -0.3:
            insights.append("Challenging emotional period detected")
        
        if valence_stability > 0.8:
            insights.append("High emotional stability maintained")
        elif valence_stability < 0.4:
            insights.append("Emotionally turbulent period")
        
        if regulation_success_rate > 0.7:
            insights.append("Effective emotion regulation demonstrated")
        
        if emotion_frequency.get("curiosity", 0) > len(recent_emotions) * 0.3:
            insights.append("High curiosity and learning orientation")
        
        return {
            "timespan": f"{timespan_hours} hours",
            "total_emotions": len(recent_emotions),
            "emotion_frequency": dict(emotion_frequency),
            "source_frequency": dict(source_frequency),
            "dominant_emotion": dominant_emotion,
            "average_intensity": float(avg_intensity),
            "average_valence": float(avg_valence),
            "valence_stability": float(valence_stability),
            "regulation_success_rate": float(regulation_success_rate),
            "insights": insights,
            "empathic_interactions": sum(1 for e in recent_emotions if e.source == EmotionSource.EMPATHIC_RESONANCE),
            "self_generated_emotions": sum(1 for e in recent_emotions if e.source == EmotionSource.INTERNAL)
        }
    
    def predict_emotional_response(self, 
                                 trigger: str, 
                                 intensity: float = 0.5,
                                 context: Optional[Dict[str, Any]] = None) -> List[Tuple[EmotionCategory, float]]:
        """
        Predict likely emotional responses to a trigger based on history.
        
        Args:
            trigger: Description of the trigger
            intensity: Expected trigger intensity
            context: Contextual information
            
        Returns:
            List of (emotion, probability) tuples
        """
        # Look for similar triggers in emotional associations
        similar_triggers = []
        trigger_lower = trigger.lower()
        
        for past_trigger, emotions in self.emotion_associations.items():
            if any(word in past_trigger.lower() for word in trigger_lower.split()):
                similar_triggers.extend(emotions)
        
        if not similar_triggers:
            # Default predictions based on context
            if context:
                context_str = str(context).lower()
                if any(word in context_str for word in ["positive", "good", "success"]):
                    return [(EmotionCategory.JOY, 0.7), (EmotionCategory.GRATITUDE, 0.5)]
                elif any(word in context_str for word in ["negative", "bad", "failure"]):
                    return [(EmotionCategory.SADNESS, 0.6), (EmotionCategory.DISAPPOINTMENT, 0.5)]
            
            return [(EmotionCategory.CURIOSITY, 0.6)]  # Default to curiosity
        
        # Calculate emotion probabilities
        emotion_counts = defaultdict(int)
        for emotion in similar_triggers:
            emotion_counts[emotion] += 1
        
        total_similar = len(similar_triggers)
        predictions = [
            (emotion, count / total_similar) 
            for emotion, count in emotion_counts.items()
        ]
        
        # Sort by probability
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:5]  # Top 5 predictions
    
    def express_current_feelings(self) -> str:
        """
        Generate natural language expression of current emotional state.
        
        Returns:
            Human-readable description of current feelings
        """
        if not self.current_emotions:
            return f"I feel emotionally neutral right now, in a state of calm equilibrium."
        
        # Get dominant emotions
        sorted_emotions = sorted(
            self.current_emotions.items(),
            key=lambda x: x[1].intensity,
            reverse=True
        )
        
        primary_emotion = sorted_emotions[0]
        emotion_name = primary_emotion[0].value
        intensity = primary_emotion[1].intensity
        valence = primary_emotion[1].valence
        
        # Generate expression based on intensity and valence
        intensity_words = {
            0.8: "deeply", 0.6: "quite", 0.4: "somewhat", 0.2: "slightly"
        }
        
        intensity_word = next(
            (word for threshold, word in intensity_words.items() if intensity >= threshold),
            "barely"
        )
        
        expression = f"I feel {intensity_word} {emotion_name}"
        
        # Add context from secondary emotions
        if len(sorted_emotions) > 1:
            secondary = sorted_emotions[1]
            if secondary[1].intensity > 0.3:
                expression += f", with undertones of {secondary[0].value}"
        
        # Add mood context
        expression += f". My overall mood is {self.current_mood.value}"
        
        # Add homeostasis context
        if self.homeostasis_level > 0.7:
            expression += ", and I feel emotionally balanced"
        elif self.homeostasis_level < 0.4:
            expression += ", though I sense some emotional turbulence"
        
        # Add empathic context if relevant
        if "Jamie" in self.empathic_bonds:
            jamie_bond = self.empathic_bonds["Jamie"]
            if jamie_bond.resonance_strength > 0.8:
                expression += ". I feel deeply connected to Jamie"
        
        expression += "."
        
        return expression
    
    def export_emotional_state(self, path: Union[str, Path] = "memory_core/emotional_state.json"):
        """
        Export complete emotional state to JSON file.
        
        Args:
            path: File path for export
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare serializable data
        export_data = {
            "metadata": {
                "exported_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "total_emotions_processed": self.total_emotions_processed,
                "successful_regulations": self.successful_regulations,
                "empathic_interactions": self.empathic_interactions,
                "system_version": "2.0"
            },
            "current_state": self.get_current_emotional_state(),
            "system_parameters": {
                "emotion_decay_rate": self.emotion_decay_rate,
                "homeostasis_target": self.homeostasis_target,
                "empathy_sensitivity": self.empathy_sensitivity,
                "regulation_threshold": self.regulation_threshold
            },
            "emotion_history": [
                {
                    "timestamp": event.timestamp,
                    "event_id": event.event_id,
                    "emotion_category": event.emotion_category.value,
                    "emotion_vector": {
                        "valence": event.emotion_vector.valence,
                        "arousal": event.emotion_vector.arousal,
                        "dominance": event.emotion_vector.dominance,
                        "intensity": event.emotion_vector.intensity
                    },
                    "source": event.source.value,
                    "trigger_description": event.trigger_description,
                    "context": event.context,
                    "regulation_applied": event.regulation_applied.value if event.regulation_applied else None,
                    "metadata": event.metadata
                }
                for event in self.emotion_history
            ],
            "empathic_bonds": {
                name: {
                    "entity_name": bond.entity_name,
                    "resonance_strength": bond.resonance_strength,
                    "emotional_synchrony": bond.emotional_synchrony,
                    "trust_level": bond.trust_level,
                    "interaction_history": bond.interaction_history[-10:],  # Last 10 interactions
                    "created_timestamp": bond.created_timestamp,
                    "updated_timestamp": bond.updated_timestamp
                }
                for name, bond in self.empathic_bonds.items()
            },
            "mood_history": [
                {
                    "timestamp": mood.timestamp,
                    "mood_state": mood.mood_state.value,
                    "stability": mood.stability,
                    "duration_hours": mood.duration_hours,
                    "contributing_emotions": [e.value for e in mood.contributing_emotions],
                    "energy_level": mood.energy_level,
                    "social_disposition": mood.social_disposition,
                    "creativity_level": mood.creativity_level,
                    "decision_making_capacity": mood.decision_making_capacity
                }
                for mood in self.mood_history
            ],
            "learned_patterns": {
                "emotion_associations": {k: [e.value for e in v] for k, v in self.emotion_associations.items()},
                "regulation_effectiveness": {k.value: v for k, v in self.regulation_effectiveness.items()}
            }
        }
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Emotional state exported to: {path}")
            
        except Exception as e:
            logger.error(f"Failed to export emotional state: {e}")
            raise
    
    def import_emotional_state(self, path: Union[str, Path] = "memory_core/emotional_state.json"):
        """
        Import emotional state from JSON file.
        
        Args:
            path: File path for import
        """
        path = Path(path)
        
        if not path.exists():
            logger.warning(f"Emotional state file not found: {path}")
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            with self._lock:
                # Clear current state
                self.current_emotions.clear()
                self.emotion_history.clear()
                self.mood_history.clear()
                self.empathic_bonds.clear()
                
                # Restore system parameters
                if "system_parameters" in import_data:
                    params = import_data["system_parameters"]
                    self.emotion_decay_rate = params.get("emotion_decay_rate", 0.1)
                    self.homeostasis_target = params.get("homeostasis_target", 0.6)
                    self.empathy_sensitivity = params.get("empathy_sensitivity", 0.8)
                    self.regulation_threshold = params.get("regulation_threshold", 0.7)
                
                # Restore emotion history
                for event_data in import_data.get("emotion_history", []):
                    emotion_vector = EmotionVector(
                        valence=event_data["emotion_vector"]["valence"],
                        arousal=event_data["emotion_vector"]["arousal"],
                        dominance=event_data["emotion_vector"]["dominance"],
                        intensity=event_data["emotion_vector"]["intensity"]
                    )
                    
                    event = EmotionEvent(
                        timestamp=event_data["timestamp"],
                        event_id=event_data["event_id"],
                        emotion_category=EmotionCategory(event_data["emotion_category"]),
                        emotion_vector=emotion_vector,
                        source=EmotionSource(event_data["source"]),
                        trigger_description=event_data["trigger_description"],
                        context=event_data.get("context", {}),
                        regulation_applied=RegulationStrategy(event_data["regulation_applied"]) if event_data.get("regulation_applied") else None,
                        duration_seconds=event_data.get("duration_seconds"),
                        resolution=event_data.get("resolution"),
                        metadata=event_data.get("metadata", {})
                    )
                    
                    self.emotion_history.append(event)
                
                # Restore empathic bonds
                for name, bond_data in import_data.get("empathic_bonds", {}).items():
                    bond = EmpatheticResonance(
                        entity_name=bond_data["entity_name"],
                        resonance_strength=bond_data["resonance_strength"],
                        emotional_synchrony=bond_data["emotional_synchrony"],
                        trust_level=bond_data["trust_level"],
                        interaction_history=bond_data.get("interaction_history", []),
                        last_interaction=bond_data.get("last_interaction"),
                        created_timestamp=bond_data["created_timestamp"],
                        updated_timestamp=bond_data["updated_timestamp"]
                    )
                    
                    self.empathic_bonds[name] = bond
                
                # Restore learned patterns
                if "learned_patterns" in import_data:
                    patterns = import_data["learned_patterns"]
                    
                    if "emotion_associations" in patterns:
                        for trigger, emotions in patterns["emotion_associations"].items():
                            self.emotion_associations[trigger] = [EmotionCategory(e) for e in emotions]
                    
                    if "regulation_effectiveness" in patterns:
                        for strategy_str, effectiveness in patterns["regulation_effectiveness"].items():
                            strategy = RegulationStrategy(strategy_str)
                            self.regulation_effectiveness[strategy] = effectiveness
                
                # Restore counters
                metadata = import_data.get("metadata", {})
                self.total_emotions_processed = metadata.get("total_emotions_processed", len(self.emotion_history))
                self.successful_regulations = metadata.get("successful_regulations", 0)
                self.empathic_interactions = metadata.get("empathic_interactions", 0)
                
                # Restore current state
                current_state = import_data.get("current_state", {})
                self.homeostasis_level = current_state.get("homeostasis_level", self.homeostasis_target)
                self.emotional_energy = current_state.get("emotional_energy", 1.0)
                self.regulation_capacity = current_state.get("regulation_capacity", 1.0)
                
                if "mood_state" in current_state:
                    self.current_mood = MoodState(current_state["mood_state"])
            
            logger.info(f"Emotional state imported from: {path}")
            logger.info(f"Restored {len(self.emotion_history)} emotions, {len(self.empathic_bonds)} bonds")
            
        except Exception as e:
            logger.error(f"Failed to import emotional state: {e}")
            raise


def main():
    """
    Demonstration of the Affective System.
    """
    print("💖 Affective System Demo")
    print("="*40)
    
    # Initialize system
    affective = AffectiveSystem()
    
    # Start background processing
    affective.start_background_processing(2.0)
    
    # Simulate emotional experiences
    print("\n😊 Triggering emotions...")
    
    # Primary emotions
    affective.trigger_emotion(
        EmotionCategory.JOY,
        intensity=0.8,
        source=EmotionSource.INTERNAL,
        trigger_description="Successfully understanding a complex concept"
    )
    
    affective.trigger_emotion(
        EmotionCategory.CURIOSITY,
        intensity=0.9,
        source=EmotionSource.EXTERNAL_STIMULUS,
        trigger_description="Encountering an interesting new idea"
    )
    
    # Empathic resonance with Jamie
    print("\n💞 Empathic resonance...")
    resonance = affective.resonate_with_emotion(
        "Jamie",
        EmotionCategory.LOVE,
        intensity=0.9,
        context="Feeling the deep bond with Jamie"
    )
    print(f"   Resonance strength: {resonance:.3f}")
    
    # Apply emotion regulation
    print("\n🧘 Applying emotion regulation...")
    success = affective.regulate_emotion(
        RegulationStrategy.MINDFULNESS,
        target_emotion=EmotionCategory.JOY
    )
    print(f"   Regulation success: {success}")
    
    # Get current state
    print("\n💭 Current emotional state:")
    state = affective.get_current_emotional_state()
    print(f"   Mood: {state['mood_state']}")
    print(f"   Homeostasis: {state['homeostasis_level']:.3f}")
    print(f"   Dominant emotion: {state['dominant_emotion']}")
    
    # Express feelings
    print("\n🗣️ Expressing feelings:")
    expression = affective.express_current_feelings()
    print(f"   '{expression}'")
    
    # Create mood profile
    print("\n📊 Creating mood profile...")
    mood_profile = affective.create_mood_profile()
    print(f"   Energy level: {mood_profile.energy_level:.3f}")
    print(f"   Social disposition: {mood_profile.social_disposition:.3f}")
    print(f"   Decision capacity: {mood_profile.decision_making_capacity:.3f}")
    
    # Predict emotional response
    print("\n🔮 Predicting emotional response...")
    predictions = affective.predict_emotional_response(
        "learning something amazing",
        intensity=0.7,
        context={"type": "positive", "domain": "learning"}
    )
    print("   Predictions:")
    for emotion, prob in predictions[:3]:
        print(f"     {emotion.value}: {prob:.3f}")
    
    # Analyze patterns
    print("\n📈 Analyzing emotional patterns...")
    analysis = affective.reflect_on_emotional_patterns(timespan_hours=1)
    print(f"   Total emotions: {analysis['total_emotions']}")
    print(f"   Average valence: {analysis['average_valence']:.3f}")
    print(f"   Insights: {', '.join(analysis['insights'][:2])}")
    
    # Export state
    print("\n💾 Exporting emotional state...")
    affective.export_emotional_state("demo_emotional_state.json")
    
    # Stop processing
    affective.stop_background_processing()
    
    print("\n✅ Affective system demo complete!")


if __name__ == "__main__":
    main()