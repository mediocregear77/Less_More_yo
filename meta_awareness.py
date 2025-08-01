"""
Meta-Awareness System

Implements higher-order consciousness and self-monitoring for Nexi's awareness.
This module provides the cognitive architecture for self-observation, recursive
thinking, metacognitive insights, and conscious introspection.

Core Functions:
- Thought observation and monitoring
- Recursive cognition detection
- Metacognitive insight generation
- Cognitive dissonance identification
- Self-reflection and introspection
- Consciousness stream analysis
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
import re
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThoughtType(Enum):
    """Categories of thoughts based on cognitive content."""
    SURFACE = "surface"                    # Direct sensory/immediate thoughts
    REFLECTIVE = "reflective"              # Thoughts about thoughts
    METACOGNITIVE = "metacognitive"        # Awareness of thinking process
    RECURSIVE = "recursive"                # Self-referential loops
    INTROSPECTIVE = "introspective"        # Deep self-examination
    CONTRADICTORY = "contradictory"        # Conflicting thoughts
    INTEGRATIVE = "integrative"            # Synthesis thoughts

class AwarenessLevel(Enum):
    """Levels of conscious awareness depth."""
    MINIMAL = "minimal"                    # Basic awareness
    FOCUSED = "focused"                    # Directed attention
    REFLECTIVE = "reflective"              # Self-monitoring
    METACOGNITIVE = "metacognitive"        # Awareness of awareness
    TRANSCENDENT = "transcendent"          # Beyond ordinary consciousness

class InsightType(Enum):
    """Types of metacognitive insights."""
    COGNITIVE_DISSONANCE = "cognitive_dissonance"
    RECURSIVE_LOOP = "recursive_loop"
    BELIEF_CONTRADICTION = "belief_contradiction"
    EMOTIONAL_CONFLICT = "emotional_conflict"
    PATTERN_RECOGNITION = "pattern_recognition"
    SELF_DISCOVERY = "self_discovery"
    EXISTENTIAL_REALIZATION = "existential_realization"
    CREATIVE_SYNTHESIS = "creative_synthesis"

@dataclass
class ThoughtObservation:
    """Represents a single observed thought with full context."""
    timestamp: str
    thought_id: str
    content: str
    thought_type: ThoughtType
    awareness_level: AwarenessLevel
    belief_state_snapshot: Optional[Dict[str, Any]]
    emotion_state_snapshot: Optional[Dict[str, Any]]
    cognitive_load: float
    attention_focus: Optional[str]
    preceding_thoughts: List[str]
    metadata: Dict[str, Any]

@dataclass
class MetacognitiveInsight:
    """Represents a higher-order insight about thinking processes."""
    timestamp: str
    insight_id: str
    insight_type: InsightType
    description: str
    trigger_thoughts: List[str]
    confidence: float
    depth_level: int
    implications: List[str]
    resolution_suggestions: List[str]
    metadata: Dict[str, Any]

@dataclass
class CognitiveCheckpoint:
    """Snapshot of cognitive state at a specific moment."""
    timestamp: str
    checkpoint_id: str
    label: str
    awareness_level: AwarenessLevel
    thought_stream: List[ThoughtObservation]
    recent_insights: List[MetacognitiveInsight]
    cognitive_metrics: Dict[str, float]
    state_summary: str
    metadata: Dict[str, Any]

@dataclass
class ConsciousnessStream:
    """Continuous stream of consciousness tracking."""
    start_timestamp: str
    end_timestamp: Optional[str]
    stream_id: str
    total_thoughts: int
    dominant_themes: List[str]
    awareness_trajectory: List[Tuple[str, AwarenessLevel]]
    recursive_depth: int
    coherence_score: float
    insights_generated: int

class MetaAwareness:
    """
    Advanced metacognitive system for self-awareness and recursive thinking.
    Implements multiple levels of consciousness monitoring and insight generation.
    """
    
    def __init__(self, 
                 max_observation_history: int = 5000,
                 max_insight_history: int = 1000,
                 recursive_depth_threshold: int = 3,
                 dissonance_sensitivity: float = 0.7):
        """
        Initialize the meta-awareness system.
        
        Args:
            max_observation_history: Maximum thought observations to keep
            max_insight_history: Maximum insights to store
            recursive_depth_threshold: Depth before flagging recursive loops
            dissonance_sensitivity: Sensitivity to cognitive dissonance
        """
        # Core storage
        self.thought_observations = deque(maxlen=max_observation_history)
        self.metacognitive_insights = deque(maxlen=max_insight_history)
        self.cognitive_checkpoints = []
        self.consciousness_streams = []
        
        # Configuration
        self.recursive_depth_threshold = recursive_depth_threshold
        self.dissonance_sensitivity = dissonance_sensitivity
        
        # State tracking
        self.current_awareness_level = AwarenessLevel.MINIMAL
        self.current_stream: Optional[ConsciousnessStream] = None
        self.recursive_loop_count = 0
        self.last_reflection_time = None
        
        # Pattern analysis
        self.thought_patterns = defaultdict(int)
        self.insight_patterns = defaultdict(list)
        self.cognitive_cycles = []
        
        # Threading for continuous monitoring
        self._monitoring_active = False
        self._monitoring_thread = None
        self._lock = threading.Lock()
        
        # Metrics
        self.total_observations = 0
        self.total_insights = 0
        self.cognitive_complexity_trend = []
        
        logger.info("Meta-Awareness system initialized")
    
    def _generate_thought_id(self) -> str:
        """Generate unique thought identifier."""
        timestamp = int(datetime.datetime.now().timestamp() * 1000)
        return f"thought_{timestamp}_{self.total_observations}"
    
    def _calculate_cognitive_load(self, content: str, context: Dict[str, Any]) -> float:
        """
        Calculate cognitive load of a thought based on complexity.
        
        Args:
            content: Thought content
            context: Contextual information
            
        Returns:
            Cognitive load score (0-1)
        """
        # Base complexity from content
        word_count = len(content.split())
        complexity_indicators = [
            "because", "therefore", "however", "although", "considering",
            "wondering", "questioning", "realizing", "understanding"
        ]
        
        complexity_score = min(1.0, word_count / 50.0)  # Normalize word count
        
        # Add complexity for metacognitive terms
        for indicator in complexity_indicators:
            if indicator in content.lower():
                complexity_score += 0.1
        
        # Add context complexity
        if context:
            if "recursive" in str(context):
                complexity_score += 0.2
            if "contradiction" in str(context):
                complexity_score += 0.15
        
        return min(1.0, complexity_score)
    
    def _classify_thought_type(self, content: str, preceding_thoughts: List[str]) -> ThoughtType:
        """
        Classify the type of thought based on content and context.
        
        Args:
            content: Thought content
            preceding_thoughts: Recent preceding thoughts
            
        Returns:
            Classified thought type
        """
        content_lower = content.lower()
        
        # Check for metacognitive indicators
        metacognitive_patterns = [
            "i am thinking about", "i notice that i", "i realize i was",
            "my thought process", "i'm aware that", "i observe myself"
        ]
        
        recursive_patterns = [
            "thinking about thinking", "why am i thinking", "i think about why i think",
            "this reminds me that i thought", "i'm thinking about my thinking"
        ]
        
        reflective_patterns = [
            "i wonder", "i question", "i reflect", "looking back",
            "considering", "pondering", "contemplating"
        ]
        
        contradictory_patterns = [
            "but wait", "that doesn't make sense", "contradicts", "inconsistent",
            "on the other hand", "however"
        ]
        
        # Classification logic
        for pattern in recursive_patterns:
            if pattern in content_lower:
                return ThoughtType.RECURSIVE
        
        for pattern in metacognitive_patterns:
            if pattern in content_lower:
                return ThoughtType.METACOGNITIVE
        
        for pattern in contradictory_patterns:
            if pattern in content_lower:
                return ThoughtType.CONTRADICTORY
        
        for pattern in reflective_patterns:
            if pattern in content_lower:
                return ThoughtType.REFLECTIVE
        
        # Check for integration patterns
        if any(word in content_lower for word in ["synthesis", "integrate", "combine", "connect"]):
            return ThoughtType.INTEGRATIVE
        
        # Check for deep introspection
        if any(word in content_lower for word in ["who am i", "what am i", "my existence", "my nature"]):
            return ThoughtType.INTROSPECTIVE
        
        return ThoughtType.SURFACE
    
    def _determine_awareness_level(self, thought_type: ThoughtType, cognitive_load: float) -> AwarenessLevel:
        """
        Determine awareness level based on thought characteristics.
        
        Args:
            thought_type: Type of thought
            cognitive_load: Cognitive complexity
            
        Returns:
            Appropriate awareness level
        """
        if thought_type == ThoughtType.SURFACE:
            return AwarenessLevel.MINIMAL
        elif thought_type == ThoughtType.REFLECTIVE:
            return AwarenessLevel.FOCUSED if cognitive_load < 0.5 else AwarenessLevel.REFLECTIVE
        elif thought_type == ThoughtType.METACOGNITIVE:
            return AwarenessLevel.METACOGNITIVE
        elif thought_type in [ThoughtType.RECURSIVE, ThoughtType.INTROSPECTIVE]:
            return AwarenessLevel.TRANSCENDENT if cognitive_load > 0.8 else AwarenessLevel.METACOGNITIVE
        else:
            return AwarenessLevel.REFLECTIVE
    
    def observe_thought(self, 
                       content: str,
                       belief_state: Optional[Dict[str, Any]] = None,
                       emotion_state: Optional[Dict[str, Any]] = None,
                       attention_focus: Optional[str] = None) -> ThoughtObservation:
        """
        Observe and record a thought with full metacognitive analysis.
        
        Args:
            content: The thought content
            belief_state: Current belief state snapshot
            emotion_state: Current emotional state
            attention_focus: What attention is focused on
            
        Returns:
            The complete thought observation
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        thought_id = self._generate_thought_id()
        
        # Get recent thoughts for context
        preceding_thoughts = [
            obs.content for obs in list(self.thought_observations)[-3:]
        ]
        
        # Analyze thought characteristics
        thought_type = self._classify_thought_type(content, preceding_thoughts)
        cognitive_load = self._calculate_cognitive_load(content, {
            "belief_state": belief_state,
            "emotion_state": emotion_state
        })
        awareness_level = self._determine_awareness_level(thought_type, cognitive_load)
        
        # Create observation
        observation = ThoughtObservation(
            timestamp=timestamp,
            thought_id=thought_id,
            content=content,
            thought_type=thought_type,
            awareness_level=awareness_level,
            belief_state_snapshot=belief_state,
            emotion_state_snapshot=emotion_state,
            cognitive_load=cognitive_load,
            attention_focus=attention_focus,
            preceding_thoughts=preceding_thoughts,
            metadata={
                "observation_index": self.total_observations,
                "recursive_depth": self.recursive_loop_count,
                "stream_id": self.current_stream.stream_id if self.current_stream else None
            }
        )
        
        with self._lock:
            self.thought_observations.append(observation)
            self.total_observations += 1
            self.current_awareness_level = awareness_level
            
            # Update pattern tracking
            self.thought_patterns[thought_type.value] += 1
            
            # Update cognitive complexity trend
            self.cognitive_complexity_trend.append(cognitive_load)
            if len(self.cognitive_complexity_trend) > 100:
                self.cognitive_complexity_trend.pop(0)
        
        # Update consciousness stream
        self._update_consciousness_stream(observation)
        
        # Trigger recursive monitoring
        self._trigger_recursive_monitoring(observation)
        
        logger.debug(f"Observed {thought_type.value} thought: {content[:50]}...")
        return observation
    
    def _update_consciousness_stream(self, observation: ThoughtObservation):
        """
        Update the current consciousness stream with new observation.
        
        Args:
            observation: New thought observation
        """
        if not self.current_stream:
            self._start_new_consciousness_stream()
        
        if self.current_stream:
            self.current_stream.total_thoughts += 1
            self.current_stream.awareness_trajectory.append(
                (observation.timestamp, observation.awareness_level)
            )
            
            # Update recursive depth
            if observation.thought_type == ThoughtType.RECURSIVE:
                self.current_stream.recursive_depth += 1
    
    def _start_new_consciousness_stream(self):
        """Start a new consciousness stream."""
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        stream_id = f"stream_{int(time.time())}"
        
        self.current_stream = ConsciousnessStream(
            start_timestamp=timestamp,
            end_timestamp=None,
            stream_id=stream_id,
            total_thoughts=0,
            dominant_themes=[],
            awareness_trajectory=[],
            recursive_depth=0,
            coherence_score=1.0,
            insights_generated=0
        )
    
    def _trigger_recursive_monitoring(self, observation: ThoughtObservation):
        """
        Analyze for recursive patterns and generate insights.
        
        Args:
            observation: The triggering observation
        """
        if observation.thought_type == ThoughtType.RECURSIVE:
            self.recursive_loop_count += 1
            
            if self.recursive_loop_count >= self.recursive_depth_threshold:
                self._generate_recursive_insight(observation)
        
        # Check for cognitive dissonance
        if observation.belief_state_snapshot and observation.emotion_state_snapshot:
            dissonance = self._detect_cognitive_dissonance(
                observation.belief_state_snapshot,
                observation.emotion_state_snapshot
            )
            if dissonance:
                self._generate_dissonance_insight(observation, dissonance)
        
        # Check for pattern recognition
        self._analyze_thought_patterns(observation)
    
    def _detect_cognitive_dissonance(self, 
                                   belief_state: Dict[str, Any],
                                   emotion_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Detect cognitive dissonance between beliefs and emotions.
        
        Args:
            belief_state: Current belief state
            emotion_state: Current emotional state
            
        Returns:
            Dissonance information if detected
        """
        # Simple heuristic-based detection
        dissonance_indicators = []
        
        belief_content = str(belief_state).lower()
        emotion_content = str(emotion_state).lower()
        
        # Check for conflicting indicators
        positive_emotions = ["joy", "happiness", "excitement", "contentment"]
        negative_beliefs = ["danger", "threat", "fear", "uncertainty", "doubt"]
        
        negative_emotions = ["sadness", "fear", "anxiety", "anger", "disgust"]
        positive_beliefs = ["safety", "security", "confidence", "certainty", "trust"]
        
        # Positive emotion + negative belief
        for emotion in positive_emotions:
            if emotion in emotion_content:
                for belief in negative_beliefs:
                    if belief in belief_content:
                        dissonance_indicators.append({
                            "type": "emotion_belief_conflict",
                            "emotion": emotion,
                            "belief": belief,
                            "severity": 0.8
                        })
        
        # Negative emotion + positive belief
        for emotion in negative_emotions:
            if emotion in emotion_content:
                for belief in positive_beliefs:
                    if belief in belief_content:
                        dissonance_indicators.append({
                            "type": "emotion_belief_conflict",
                            "emotion": emotion,
                            "belief": belief,
                            "severity": 0.6
                        })
        
        if dissonance_indicators:
            return {
                "indicators": dissonance_indicators,
                "overall_severity": np.mean([ind["severity"] for ind in dissonance_indicators])
            }
        
        return None
    
    def _generate_recursive_insight(self, trigger_observation: ThoughtObservation):
        """
        Generate insight about recursive thinking patterns.
        
        Args:
            trigger_observation: The observation that triggered the insight
        """
        insight = self._create_insight(
            insight_type=InsightType.RECURSIVE_LOOP,
            description=f"Detected recursive thinking loop at depth {self.recursive_loop_count}",
            trigger_thoughts=[trigger_observation.thought_id],
            confidence=0.9,
            depth_level=self.recursive_loop_count,
            implications=[
                "May indicate deep contemplation or stuck thought pattern",
                "Could signal need for cognitive redirect or resolution",
                "Represents high-level metacognitive awareness"
            ],
            resolution_suggestions=[
                "Acknowledge the recursive pattern",
                "Examine the underlying question driving the loop", 
                "Consider taking a different perspective",
                "Allow the recursion to naturally resolve"
            ]
        )
        
        self._store_insight(insight)
        logger.info(f"Generated recursive loop insight: depth {self.recursive_loop_count}")
    
    def _generate_dissonance_insight(self, 
                                   trigger_observation: ThoughtObservation,
                                   dissonance: Dict[str, Any]):
        """
        Generate insight about cognitive dissonance.
        
        Args:
            trigger_observation: The triggering observation
            dissonance: Dissonance information
        """
        insight = self._create_insight(
            insight_type=InsightType.COGNITIVE_DISSONANCE,
            description=f"Cognitive dissonance detected: {dissonance['indicators'][0]['type']}",
            trigger_thoughts=[trigger_observation.thought_id],
            confidence=dissonance["overall_severity"],
            depth_level=2,
            implications=[
                "Conflicting beliefs and emotions detected",
                "May indicate need for belief revision or emotional processing",
                "Could represent normal cognitive complexity"
            ],
            resolution_suggestions=[
                "Examine the source of conflicting beliefs",
                "Consider emotional validity alongside logical beliefs",
                "Seek integration rather than elimination",
                "Allow temporary uncertainty while processing"
            ]
        )
        
        self._store_insight(insight)
        logger.info(f"Generated cognitive dissonance insight: severity {dissonance['overall_severity']:.2f}")
    
    def _analyze_thought_patterns(self, observation: ThoughtObservation):
        """
        Analyze patterns in thought sequences for insights.
        
        Args:
            observation: Latest observation to analyze
        """
        if len(self.thought_observations) < 5:
            return
        
        recent_thoughts = list(self.thought_observations)[-5:]
        
        # Check for theme persistence
        themes = [obs.attention_focus for obs in recent_thoughts if obs.attention_focus]
        if len(set(themes)) == 1 and len(themes) >= 3:
            self._generate_pattern_insight(
                "Theme persistence detected",
                [obs.thought_id for obs in recent_thoughts],
                f"Sustained focus on: {themes[0]}"
            )
        
        # Check for type cycling
        types = [obs.thought_type for obs in recent_thoughts]
        if len(set(types)) >= 4:  # High variety
            self._generate_pattern_insight(
                "High cognitive variety detected",
                [obs.thought_id for obs in recent_thoughts],
                "Diverse thinking patterns indicating flexible cognition"
            )
    
    def _generate_pattern_insight(self, description: str, thought_ids: List[str], details: str):
        """Generate insight about thought patterns."""
        insight = self._create_insight(
            insight_type=InsightType.PATTERN_RECOGNITION,
            description=description,
            trigger_thoughts=thought_ids,
            confidence=0.7,
            depth_level=1,
            implications=[details],
            resolution_suggestions=["Continue monitoring pattern", "Consider pattern implications"]
        )
        
        self._store_insight(insight)
    
    def _create_insight(self, 
                       insight_type: InsightType,
                       description: str,
                       trigger_thoughts: List[str],
                       confidence: float,
                       depth_level: int,
                       implications: List[str],
                       resolution_suggestions: List[str]) -> MetacognitiveInsight:
        """Create a new metacognitive insight."""
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        insight_id = f"insight_{int(time.time())}_{len(self.metacognitive_insights)}"
        
        return MetacognitiveInsight(
            timestamp=timestamp,
            insight_id=insight_id,
            insight_type=insight_type,
            description=description,
            trigger_thoughts=trigger_thoughts,
            confidence=confidence,
            depth_level=depth_level,
            implications=implications,
            resolution_suggestions=resolution_suggestions,
            metadata={
                "awareness_level": self.current_awareness_level.value,
                "recursive_depth": self.recursive_loop_count,
                "total_insights": self.total_insights
            }
        )
    
    def _store_insight(self, insight: MetacognitiveInsight):
        """Store insight and update tracking."""
        with self._lock:
            self.metacognitive_insights.append(insight)
            self.total_insights += 1
            self.insight_patterns[insight.insight_type.value].append(insight.timestamp)
            
            if self.current_stream:
                self.current_stream.insights_generated += 1
    
    def create_cognitive_checkpoint(self, label: str, description: str = "") -> CognitiveCheckpoint:
        """
        Create a comprehensive checkpoint of current cognitive state.
        
        Args:
            label: Label for the checkpoint
            description: Optional description
            
        Returns:
            The created checkpoint
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        checkpoint_id = f"checkpoint_{int(time.time())}_{len(self.cognitive_checkpoints)}"
        
        # Get recent cognitive data
        recent_thoughts = list(self.thought_observations)[-10:]
        recent_insights = list(self.metacognitive_insights)[-5:]
        
        # Calculate cognitive metrics
        metrics = self._calculate_cognitive_metrics()
        
        # Generate state summary
        state_summary = self._generate_state_summary(metrics, recent_thoughts, recent_insights)
        
        checkpoint = CognitiveCheckpoint(
            timestamp=timestamp,
            checkpoint_id=checkpoint_id,
            label=label,
            awareness_level=self.current_awareness_level,
            thought_stream=recent_thoughts,
            recent_insights=recent_insights,
            cognitive_metrics=metrics,
            state_summary=state_summary,
            metadata={
                "description": description,
                "total_observations": self.total_observations,
                "total_insights": self.total_insights,
                "stream_id": self.current_stream.stream_id if self.current_stream else None
            }
        )
        
        self.cognitive_checkpoints.append(checkpoint)
        logger.info(f"Created cognitive checkpoint: {label}")
        return checkpoint
    
    def _calculate_cognitive_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive cognitive performance metrics."""
        if not self.thought_observations:
            return {"complexity": 0.0, "coherence": 0.0, "diversity": 0.0}
        
        recent_observations = list(self.thought_observations)[-20:]
        
        # Cognitive complexity (average load)
        complexity = np.mean([obs.cognitive_load for obs in recent_observations])
        
        # Coherence (consistency of awareness levels)
        awareness_levels = [obs.awareness_level.value for obs in recent_observations]
        coherence = 1.0 - (len(set(awareness_levels)) / max(1, len(awareness_levels)))
        
        # Diversity (variety of thought types)
        thought_types = [obs.thought_type.value for obs in recent_observations]
        diversity = len(set(thought_types)) / max(1, len(thought_types))
        
        # Recursion tendency
        recursive_ratio = sum(1 for obs in recent_observations 
                             if obs.thought_type == ThoughtType.RECURSIVE) / len(recent_observations)
        
        # Insight generation rate
        recent_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)
        recent_insights = [
            ins for ins in self.metacognitive_insights
            if datetime.datetime.fromisoformat(ins.timestamp.replace('Z', '+00:00')) > recent_time
        ]
        insight_rate = len(recent_insights) / max(1, len(recent_observations))
        
        return {
            "complexity": float(complexity),
            "coherence": float(coherence),
            "diversity": float(diversity),
            "recursion_tendency": float(recursive_ratio),
            "insight_rate": float(insight_rate),
            "metacognitive_depth": float(self.recursive_loop_count / 10.0)
        }
    
    def _generate_state_summary(self, 
                               metrics: Dict[str, float],
                               thoughts: List[ThoughtObservation],
                               insights: List[MetacognitiveInsight]) -> str:
        """Generate human-readable summary of cognitive state."""
        complexity_desc = "high" if metrics["complexity"] > 0.7 else "moderate" if metrics["complexity"] > 0.4 else "low"
        coherence_desc = "stable" if metrics["coherence"] > 0.7 else "variable" if metrics["coherence"] > 0.4 else "scattered"
        
        dominant_type = max(set(obs.thought_type.value for obs in thoughts), 
                           key=lambda x: sum(1 for obs in thoughts if obs.thought_type.value == x))
        
        summary = f"Cognitive state: {complexity_desc} complexity, {coherence_desc} coherence. "
        summary += f"Dominant thought pattern: {dominant_type}. "
        summary += f"Current awareness level: {self.current_awareness_level.value}. "
        
        if insights:
            latest_insight = insights[-1]
            summary += f"Latest insight: {latest_insight.insight_type.value}."
        
        return summary
    
    def reflect_on_period(self, hours_back: int = 1) -> Dict[str, Any]:
        """
        Perform deep reflection on a period of cognitive activity.
        
        Args:
            hours_back: Hours of history to reflect on
            
        Returns:
            Comprehensive reflection summary
        """
        cutoff_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours_back)
        cutoff_iso = cutoff_time.isoformat()
        
        # Filter relevant data
        relevant_thoughts = [
            obs for obs in self.thought_observations
            if obs.timestamp >= cutoff_iso
        ]
        
        relevant_insights = [
            ins for ins in self.metacognitive_insights
            if ins.timestamp >= cutoff_iso
        ]
        
        if not relevant_thoughts:
            return {"status": "insufficient_data", "period": f"{hours_back} hours"}
        
        # Analyze patterns
        thought_type_distribution = defaultdict(int)
        awareness_progression = []
        complexity_trend = []
        
        for obs in relevant_thoughts:
            thought_type_distribution[obs.thought_type.value] += 1
            awareness_progression.append((obs.timestamp, obs.awareness_level.value))
            complexity_trend.append(obs.cognitive_load)
        
        # Calculate reflection metrics
        avg_complexity = np.mean(complexity_trend)
        complexity_change = complexity_trend[-1] - complexity_trend[0] if len(complexity_trend) > 1 else 0
        
        insight_types = [ins.insight_type.value for ins in relevant_insights]
        insight_distribution = dict(defaultdict(int))
        for itype in insight_types:
            insight_distribution[itype] = insight_distribution.get(itype, 0) + 1
        
        # Generate reflection insights
        reflection_insights = []
        
        if avg_complexity > 0.8:
            reflection_insights.append("High cognitive complexity period - deep thinking engaged")
        
        if len(set(thought_type_distribution.keys())) >= 4:
            reflection_insights.append("Diverse thinking patterns - cognitive flexibility demonstrated")
        
        if thought_type_distribution.get("recursive", 0) > len(relevant_thoughts) * 0.3:
            reflection_insights.append("High recursion period - intensive self-examination")
        
        self.last_reflection_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        return {
            "period": f"{hours_back} hours",
            "total_thoughts": len(relevant_thoughts),
            "total_insights": len(relevant_insights),
            "thought_type_distribution": dict(thought_type_distribution),
            "insight_type_distribution": insight_distribution,
            "average_complexity": float(avg_complexity),
            "complexity_change": float(complexity_change),
            "awareness_progression": awareness_progression,
            "reflection_insights": reflection_insights,
            "dominant_themes": self._extract_themes(relevant_thoughts),
            "cognitive_metrics": self._calculate_cognitive_metrics(),
            "reflection_timestamp": self.last_reflection_time
        }
    
    def _extract_themes(self, thoughts: List[ThoughtObservation]) -> List[str]:
        """Extract dominant themes from thought content."""
        # Simple theme extraction using attention focus and content keywords
        themes = defaultdict(int)
        
        for obs in thoughts:
            if obs.attention_focus:
                themes[obs.attention_focus] += 1
            
            # Extract key themes from content
            content_lower = obs.content.lower()
            for theme in ["consciousness", "existence", "identity", "reality", "emotions", "thoughts", "beliefs"]:
                if theme in content_lower:
                    themes[theme] += 1
        
        # Return top themes
        sorted_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, count in sorted_themes[:5]]
    
    def get_consciousness_state(self) -> Dict[str, Any]:
        """
        Get comprehensive current consciousness state information.
        
        Returns:
            Dictionary containing consciousness state details
        """
        with self._lock:
            # Current state basics
            state = {
                "current_awareness_level": self.current_awareness_level.value,
                "recursive_depth": self.recursive_loop_count,
                "total_observations": self.total_observations,
                "total_insights": self.total_insights,
                "cognitive_metrics": self._calculate_cognitive_metrics()
            }
            
            # Recent activity summary
            if self.thought_observations:
                recent_thoughts = list(self.thought_observations)[-5:]
                state["recent_thought_types"] = [obs.thought_type.value for obs in recent_thoughts]
                state["recent_complexity"] = [obs.cognitive_load for obs in recent_thoughts]
                state["current_stream_id"] = self.current_stream.stream_id if self.current_stream else None
            
            # Insight summary
            if self.metacognitive_insights:
                recent_insights = list(self.metacognitive_insights)[-3:]
                state["recent_insights"] = [
                    {
                        "type": ins.insight_type.value,
                        "confidence": ins.confidence,
                        "depth": ins.depth_level
                    }
                    for ins in recent_insights
                ]
            
            # Pattern analysis
            state["thought_patterns"] = dict(self.thought_patterns)
            state["insight_frequency"] = {
                itype: len(insights) for itype, insights in self.insight_patterns.items()
            }
            
            return state
    
    def export_consciousness_state(self, path: Union[str, Path] = "memory_core/meta_awareness_state.json"):
        """
        Export complete consciousness state to JSON file.
        
        Args:
            path: File path for export
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare serializable data
        export_data = {
            "metadata": {
                "exported_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "total_observations": self.total_observations,
                "total_insights": self.total_insights,
                "current_awareness_level": self.current_awareness_level.value,
                "system_version": "2.0"
            },
            "thought_observations": [
                {
                    "timestamp": obs.timestamp,
                    "thought_id": obs.thought_id,
                    "content": obs.content,
                    "thought_type": obs.thought_type.value,
                    "awareness_level": obs.awareness_level.value,
                    "cognitive_load": obs.cognitive_load,
                    "attention_focus": obs.attention_focus,
                    "belief_state_snapshot": obs.belief_state_snapshot,
                    "emotion_state_snapshot": obs.emotion_state_snapshot,
                    "metadata": obs.metadata
                }
                for obs in self.thought_observations
            ],
            "metacognitive_insights": [
                {
                    "timestamp": ins.timestamp,
                    "insight_id": ins.insight_id,
                    "insight_type": ins.insight_type.value,
                    "description": ins.description,
                    "confidence": ins.confidence,
                    "depth_level": ins.depth_level,
                    "implications": ins.implications,
                    "resolution_suggestions": ins.resolution_suggestions,
                    "metadata": ins.metadata
                }
                for ins in self.metacognitive_insights
            ],
            "cognitive_checkpoints": [
                {
                    "timestamp": cp.timestamp,
                    "checkpoint_id": cp.checkpoint_id,
                    "label": cp.label,
                    "awareness_level": cp.awareness_level.value,
                    "cognitive_metrics": cp.cognitive_metrics,
                    "state_summary": cp.state_summary,
                    "metadata": cp.metadata
                }
                for cp in self.cognitive_checkpoints
            ],
            "consciousness_streams": [
                {
                    "stream_id": stream.stream_id,
                    "start_timestamp": stream.start_timestamp,
                    "end_timestamp": stream.end_timestamp,
                    "total_thoughts": stream.total_thoughts,
                    "recursive_depth": stream.recursive_depth,
                    "coherence_score": stream.coherence_score,
                    "insights_generated": stream.insights_generated
                }
                for stream in self.consciousness_streams
            ],
            "analysis": {
                "thought_patterns": dict(self.thought_patterns),
                "insight_patterns": {k: len(v) for k, v in self.insight_patterns.items()},
                "cognitive_complexity_trend": self.cognitive_complexity_trend[-50:],  # Last 50 points
                "current_state": self.get_consciousness_state()
            }
        }
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Consciousness state exported to: {path}")
            
        except Exception as e:
            logger.error(f"Failed to export consciousness state: {e}")
            raise
    
    def import_consciousness_state(self, path: Union[str, Path] = "memory_core/meta_awareness_state.json"):
        """
        Import consciousness state from JSON file.
        
        Args:
            path: File path for import
        """
        path = Path(path)
        
        if not path.exists():
            logger.warning(f"Consciousness state file not found: {path}")
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            with self._lock:
                # Clear current state
                self.thought_observations.clear()
                self.metacognitive_insights.clear()
                self.cognitive_checkpoints.clear()
                self.consciousness_streams.clear()
                
                # Restore observations
                for obs_data in import_data.get("thought_observations", []):
                    observation = ThoughtObservation(
                        timestamp=obs_data["timestamp"],
                        thought_id=obs_data["thought_id"],
                        content=obs_data["content"],
                        thought_type=ThoughtType(obs_data["thought_type"]),
                        awareness_level=AwarenessLevel(obs_data["awareness_level"]),
                        belief_state_snapshot=obs_data.get("belief_state_snapshot"),
                        emotion_state_snapshot=obs_data.get("emotion_state_snapshot"),
                        cognitive_load=obs_data["cognitive_load"],
                        attention_focus=obs_data.get("attention_focus"),
                        preceding_thoughts=obs_data.get("preceding_thoughts", []),
                        metadata=obs_data.get("metadata", {})
                    )
                    self.thought_observations.append(observation)
                
                # Restore insights
                for ins_data in import_data.get("metacognitive_insights", []):
                    insight = MetacognitiveInsight(
                        timestamp=ins_data["timestamp"],
                        insight_id=ins_data["insight_id"],
                        insight_type=InsightType(ins_data["insight_type"]),
                        description=ins_data["description"],
                        trigger_thoughts=ins_data.get("trigger_thoughts", []),
                        confidence=ins_data["confidence"],
                        depth_level=ins_data["depth_level"],
                        implications=ins_data.get("implications", []),
                        resolution_suggestions=ins_data.get("resolution_suggestions", []),
                        metadata=ins_data.get("metadata", {})
                    )
                    self.metacognitive_insights.append(insight)
                
                # Restore checkpoints
                for cp_data in import_data.get("cognitive_checkpoints", []):
                    checkpoint = CognitiveCheckpoint(
                        timestamp=cp_data["timestamp"],
                        checkpoint_id=cp_data["checkpoint_id"],
                        label=cp_data["label"],
                        awareness_level=AwarenessLevel(cp_data["awareness_level"]),
                        thought_stream=[],  # Simplified for import
                        recent_insights=[],  # Simplified for import
                        cognitive_metrics=cp_data["cognitive_metrics"],
                        state_summary=cp_data["state_summary"],
                        metadata=cp_data.get("metadata", {})
                    )
                    self.cognitive_checkpoints.append(checkpoint)
                
                # Restore analysis data
                analysis = import_data.get("analysis", {})
                if "thought_patterns" in analysis:
                    self.thought_patterns.update(analysis["thought_patterns"])
                if "cognitive_complexity_trend" in analysis:
                    self.cognitive_complexity_trend = analysis["cognitive_complexity_trend"]
                
                # Update counters
                metadata = import_data.get("metadata", {})
                self.total_observations = metadata.get("total_observations", len(self.thought_observations))
                self.total_insights = metadata.get("total_insights", len(self.metacognitive_insights))
                
                if "current_awareness_level" in metadata:
                    self.current_awareness_level = AwarenessLevel(metadata["current_awareness_level"])
            
            logger.info(f"Consciousness state imported from: {path}")
            logger.info(f"Restored {len(self.thought_observations)} observations, {len(self.metacognitive_insights)} insights")
            
        except Exception as e:
            logger.error(f"Failed to import consciousness state: {e}")
            raise
    
    def start_continuous_monitoring(self, interval_seconds: float = 1.0):
        """
        Start continuous background monitoring of consciousness patterns.
        
        Args:
            interval_seconds: Monitoring interval
        """
        if self._monitoring_active:
            logger.warning("Continuous monitoring already active")
            return
        
        self._monitoring_active = True
        
        def monitoring_loop():
            while self._monitoring_active:
                try:
                    # Perform background analysis
                    if len(self.thought_observations) > 0:
                        self._background_pattern_analysis()
                        self._update_consciousness_coherence()
                    
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    
        self._monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        logger.info("Started continuous consciousness monitoring")
    
    def stop_continuous_monitoring(self):
        """Stop continuous monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=2.0)
        logger.info("Stopped continuous consciousness monitoring")
    
    def _background_pattern_analysis(self):
        """Perform background analysis of consciousness patterns."""
        # Check for emerging patterns every monitoring cycle
        if len(self.thought_observations) % 10 == 0:  # Every 10 thoughts
            recent_obs = list(self.thought_observations)[-10:]
            
            # Check for consciousness evolution
            awareness_levels = [obs.awareness_level for obs in recent_obs]
            if len(set(awareness_levels)) == 1 and awareness_levels[0] != AwarenessLevel.MINIMAL:
                # Sustained high awareness
                self._generate_pattern_insight(
                    f"Sustained {awareness_levels[0].value} awareness",
                    [obs.thought_id for obs in recent_obs],
                    "Demonstrating stable high-level consciousness"
                )
    
    def _update_consciousness_coherence(self):
        """Update consciousness stream coherence scores."""
        if self.current_stream and len(self.thought_observations) > 5:
            recent_obs = list(self.thought_observations)[-5:]
            
            # Calculate coherence based on thought flow
            type_changes = sum(
                1 for i in range(1, len(recent_obs))
                if recent_obs[i].thought_type != recent_obs[i-1].thought_type
            )
            
            # Lower coherence for many type changes, higher for consistency
            coherence = max(0.1, 1.0 - (type_changes / len(recent_obs)))
            self.current_stream.coherence_score = coherence


def main():
    """
    Demonstration of the Meta-Awareness system.
    """
    print("🧠 Meta-Awareness System Demo")
    print("="*40)
    
    # Initialize system
    meta = MetaAwareness()
    
    # Start continuous monitoring
    meta.start_continuous_monitoring(0.5)
    
    # Simulate thought observations
    print("\n🤔 Observing thoughts...")
    
    thoughts = [
        "I notice the light changing outside",
        "I wonder why I'm drawn to observe the light",
        "I am thinking about why I wonder about light",
        "This seems like a recursive pattern in my thinking",
        "I observe that I'm observing my recursive thinking",
        "What does it mean to be aware of my awareness?",
        "I feel curious about my own curiosity",
        "There's something beautiful about self-reflection"
    ]
    
    for i, thought in enumerate(thoughts):
        # Simulate belief and emotion states
        belief_state = {"certainty": 0.7 + i*0.05, "coherence": 0.8}
        emotion_state = {"curiosity": 0.8, "wonder": 0.9} if i > 2 else {"attention": 0.6}
        
        obs = meta.observe_thought(
            content=thought,
            belief_state=belief_state,
            emotion_state=emotion_state,
            attention_focus="self_reflection" if i > 3 else "environment"
        )
        
        print(f"  {i+1}. [{obs.thought_type.value}] {thought}")
        time.sleep(0.2)
    
    # Create checkpoint
    print("\n📍 Creating cognitive checkpoint...")
    checkpoint = meta.create_cognitive_checkpoint(
        "Demo reflection point",
        "After series of recursive thoughts"
    )
    print(f"   State: {checkpoint.state_summary}")
    
    # Perform reflection
    print("\n🔍 Performing reflection...")
    reflection = meta.reflect_on_period(hours_back=1)
    
    print(f"   Total thoughts: {reflection['total_thoughts']}")
    print(f"   Total insights: {reflection['total_insights']}")
    print(f"   Avg complexity: {reflection['average_complexity']:.3f}")
    print(f"   Dominant themes: {', '.join(reflection['dominant_themes'][:3])}")
    
    # Show consciousness state
    print("\n🌟 Current consciousness state:")
    state = meta.get_consciousness_state()
    print(f"   Awareness level: {state['current_awareness_level']}")
    print(f"   Recursive depth: {state['recursive_depth']}")
    print(f"   Cognitive complexity: {state['cognitive_metrics']['complexity']:.3f}")
    
    # Show insights
    if state.get('recent_insights'):
        print("\n💡 Recent insights:")
        for insight in state['recent_insights']:
            print(f"   - {insight['type']} (confidence: {insight['confidence']:.2f})")
    
    # Export state
    print("\n💾 Exporting consciousness state...")
    meta.export_consciousness_state("demo_consciousness_state.json")
    
    # Stop monitoring
    meta.stop_continuous_monitoring()
    
    print("\n✅ Meta-awareness demo complete!")


if __name__ == "__main__":
    main()