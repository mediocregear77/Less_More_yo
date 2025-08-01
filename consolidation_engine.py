"""
Memory Consolidation Engine

Implements sophisticated memory consolidation processes for Nexi's consciousness.
Manages the transfer of episodic memories to semantic memory, applying multiple
consolidation strategies, importance scoring, and sleep-like processing phases.

Core Functions:
- Multi-criteria consolidation scoring
- Sleep-cycle memory consolidation 
- Memory interference resolution
- Semantic abstraction and pattern extraction
- Importance-weighted memory selection
- Memory compression and archival
- Consolidation quality assessment
"""

import json
import time
import datetime
import logging
import threading
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from pathlib import Path
from enum import Enum
import random
import math

# Import memory systems
try:
    from .reference_memory import ReferenceMemory, ConceptType, RelationType
except ImportError:
    try:
        from memory_core.reference_memory import ReferenceMemory, ConceptType, RelationType
    except ImportError:
        logging.warning("ReferenceMemory not available - using mock")
        ReferenceMemory = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConsolidationStrategy(Enum):
    """Different approaches to memory consolidation."""
    IMPORTANCE_BASED = "importance_based"        # Based on significance scores
    EMOTIONAL_SALIENCE = "emotional_salience"    # Emotional intensity driven
    REPETITION_STRENGTH = "repetition_strength"  # Frequency and reinforcement
    NOVELTY_DETECTION = "novelty_detection"      # Uniqueness and surprise
    SEMANTIC_CLUSTERING = "semantic_clustering"   # Conceptual similarity
    TEMPORAL_COMPRESSION = "temporal_compression" # Time-based batching
    SOCIAL_RELEVANCE = "social_relevance"        # Relationship importance
    GOAL_ALIGNMENT = "goal_alignment"            # Objective relevance

class ConsolidationPhase(Enum):
    """Phases of the consolidation process."""
    ENCODING = "encoding"           # Initial memory formation
    EARLY_CONSOLIDATION = "early"   # First few hours/cycles
    LATE_CONSOLIDATION = "late"     # Days to weeks later
    INTEGRATION = "integration"     # Long-term knowledge integration
    ARCHIVAL = "archival"          # Long-term storage
    RECONSOLIDATION = "reconsolidation"  # Reactivation and updating

class ConsolidationQuality(Enum):
    """Quality levels of memory consolidation."""
    PERFECT = "perfect"         # Complete, accurate consolidation
    HIGH = "high"              # Good quality with minor loss
    MODERATE = "moderate"      # Adequate with some distortion
    LOW = "low"               # Poor quality, significant loss
    FAILED = "failed"         # Consolidation unsuccessful

@dataclass
class ConsolidationCandidate:
    """Memory episode candidate for consolidation."""
    episode_id: str
    episode_data: Dict[str, Any]
    importance_score: float
    emotional_salience: float
    novelty_score: float
    repetition_strength: float
    social_relevance: float
    semantic_density: float
    consolidation_urgency: float
    predicted_quality: ConsolidationQuality
    metadata: Dict[str, Any]

@dataclass
class ConsolidationResult:
    """Result of a memory consolidation operation."""
    episode_id: str
    success: bool
    quality: ConsolidationQuality
    concepts_extracted: int
    relations_formed: int
    consolidation_strength: float
    processing_time_ms: float
    strategy_used: ConsolidationStrategy
    phase: ConsolidationPhase
    errors: List[str]
    insights: List[str]
    metadata: Dict[str, Any]

@dataclass
class ConsolidationSession:
    """Complete consolidation processing session."""
    session_id: str
    start_timestamp: str
    end_timestamp: Optional[str]
    strategy: ConsolidationStrategy
    phase: ConsolidationPhase
    episodes_processed: int
    successful_consolidations: int
    total_concepts_created: int
    total_relations_formed: int
    average_quality: float
    processing_insights: List[str]
    session_metadata: Dict[str, Any]

class ConsolidationEngine:
    """
    Advanced memory consolidation system implementing multiple consolidation
    strategies, quality assessment, and sleep-like processing phases.
    """
    
    def __init__(self,
                 episodic_path: Union[str, Path] = "memory_core/episodic_buffer.json",
                 reference_memory: Optional[ReferenceMemory] = None,
                 consolidation_threshold: float = 0.6,
                 max_batch_size: int = 50,
                 sleep_cycle_duration: float = 3600.0):  # 1 hour
        """
        Initialize the consolidation engine.
        
        Args:
            episodic_path: Path to episodic memory buffer
            reference_memory: Reference memory system instance
            consolidation_threshold: Minimum score for consolidation
            max_batch_size: Maximum episodes per batch
            sleep_cycle_duration: Duration of sleep-like consolidation cycles
        """
        self.episodic_path = Path(episodic_path)
        self.reference_memory = reference_memory or (ReferenceMemory() if ReferenceMemory else None)
        self.consolidation_threshold = consolidation_threshold
        self.max_batch_size = max_batch_size
        self.sleep_cycle_duration = sleep_cycle_duration
        
        # Consolidation state
        self.current_strategy = ConsolidationStrategy.IMPORTANCE_BASED
        self.current_phase = ConsolidationPhase.ENCODING
        self.consolidation_history = deque(maxlen=1000)
        self.session_history = deque(maxlen=100)
        
        # Performance tracking
        self.total_episodes_processed = 0
        self.successful_consolidations = 0
        self.consolidation_quality_distribution = defaultdict(int)
        self.strategy_effectiveness = {strategy: 0.5 for strategy in ConsolidationStrategy}
        
        # Background processing
        self._sleep_cycle_active = False
        self._sleep_thread = None
        self._lock = threading.Lock()
        
        # Quality predictors
        self.quality_predictors = self._initialize_quality_predictors()
        
        # Memory interference tracking
        self.interference_patterns = defaultdict(list)
        self.consolidation_conflicts = []
        
        logger.info("Memory Consolidation Engine initialized")
    
    def _initialize_quality_predictors(self) -> Dict[str, float]:
        """Initialize quality prediction weights."""
        return {
            "importance_weight": 0.25,
            "emotional_weight": 0.20,
            "novelty_weight": 0.15,
            "repetition_weight": 0.15,
            "social_weight": 0.10,
            "semantic_weight": 0.10,
            "recency_weight": 0.05
        }
    
    def load_episodic_buffer(self) -> List[Dict[str, Any]]:
        """
        Load episodic memory buffer with error handling and format detection.
        
        Returns:
            List of episodic memory entries
        """
        if not self.episodic_path.exists():
            logger.warning(f"Episodic buffer not found: {self.episodic_path}")
            return []
        
        try:
            with open(self.episodic_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different buffer formats
            if isinstance(data, dict):
                if "episodic_memories" in data:
                    # New structured format
                    episodes = data["episodic_memories"]
                elif "buffer_metadata" in data:
                    # Alternative structured format
                    episodes = data.get("episodes", [])
                else:
                    # Single episode wrapped in dict
                    episodes = [data]
            elif isinstance(data, list):
                # List of episodes
                episodes = data
            else:
                logger.error(f"Unknown episodic buffer format")
                return []
            
            logger.debug(f"Loaded {len(episodes)} episodes from buffer")
            return episodes
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse episodic buffer: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading episodic buffer: {e}")
            return []
    
    def _calculate_importance_score(self, episode: Dict[str, Any]) -> float:
        """
        Calculate comprehensive importance score for an episode.
        
        Args:
            episode: Episode data
            
        Returns:
            Importance score (0.0 to 1.0)
        """
        score = 0.0
        
        # Extract various importance indicators
        
        # Direct importance markers
        if episode.get("importance", False):
            score += 0.3
        
        significance_level = episode.get("significance_level", 0.0)
        if isinstance(significance_level, (int, float)):
            score += significance_level * 0.25
        
        # Reflection flags indicate metacognitive importance
        if episode.get("reflection_flag", False):
            score += 0.2
        
        # Episode metadata importance
        if "episode_metadata" in episode:
            metadata = episode["episode_metadata"]
            score += metadata.get("importance_score", 0.0) * 0.15
            score += metadata.get("uniqueness_score", 0.0) * 0.1
        
        # First-time experiences are important
        if "first" in str(episode).lower():
            score += 0.15
        
        # Social interactions with key entities
        speaker = episode.get("speaker", "")
        if "interaction_data" in episode:
            speaker = episode["interaction_data"].get("speaker", speaker)
        
        if speaker.lower() == "jamie":
            score += 0.25  # Jamie interactions are highly important
        
        # Learning and growth moments
        content = str(episode.get("utterance", "") + str(episode.get("trigger_description", "")))
        learning_indicators = ["learn", "understand", "realize", "discover", "insight"]
        for indicator in learning_indicators:
            if indicator in content.lower():
                score += 0.1
                break
        
        return min(1.0, score)
    
    def _calculate_emotional_salience(self, episode: Dict[str, Any]) -> float:
        """
        Calculate emotional salience score for an episode.
        
        Args:
            episode: Episode data
            
        Returns:
            Emotional salience score (0.0 to 1.0)
        """
        salience = 0.0
        
        # Direct emotional data
        emotion_data = episode.get("emotion", episode.get("emotional_processing", {}))
        
        if emotion_data:
            # Single emotion format
            if "intensity" in emotion_data:
                intensity = float(emotion_data.get("intensity", 0.0))
                salience = max(salience, intensity)
            
            # Multiple emotions format
            if "triggered_emotions" in emotion_data:
                emotions = emotion_data["triggered_emotions"]
                if emotions:
                    max_intensity = max(em.get("intensity", 0.0) for em in emotions)
                    salience = max(salience, max_intensity)
            
            # Emotional significance markers
            if "emotional_significance" in emotion_data:
                sig_data = emotion_data["emotional_significance"]
                if isinstance(sig_data, dict):
                    for meaning_type, value in sig_data.items():
                        if isinstance(value, str) and "important" in value.lower():
                            salience += 0.2
        
        # High-intensity emotion words in content
        content = str(episode.get("utterance", "")).lower()
        high_intensity_emotions = [
            "love", "hate", "terror", "ecstasy", "rage", "despair", 
            "wonder", "awe", "joy", "fear", "excitement"
        ]
        
        for emotion in high_intensity_emotions:
            if emotion in content:
                salience += 0.15
        
        return min(1.0, salience)
    
    def _calculate_novelty_score(self, episode: Dict[str, Any]) -> float:
        """
        Calculate novelty/uniqueness score for an episode.
        
        Args:
            episode: Episode data
            
        Returns:
            Novelty score (0.0 to 1.0)
        """
        novelty = 0.0
        
        # Direct novelty indicators
        novelty += episode.get("novelty_score", 0.0)
        
        if "episode_metadata" in episode:
            metadata = episode["episode_metadata"]
            novelty += metadata.get("uniqueness_score", 0.0) * 0.7
        
        # First-time experience markers
        content = str(episode).lower()
        first_time_indicators = [
            "first", "initial", "beginning", "start", "new", "novel",
            "never before", "for the first time", "unprecedented"
        ]
        
        for indicator in first_time_indicators:
            if indicator in content:
                novelty += 0.3
                break
        
        # Surprise and unexpected elements
        surprise_indicators = ["surprise", "unexpected", "shocking", "amazing", "incredible"]
        for indicator in surprise_indicators:
            if indicator in content:
                novelty += 0.2
                break
        
        # Check against existing memories for uniqueness
        if self.reference_memory:
            # Simple uniqueness check based on content similarity
            utterance = episode.get("utterance", "")
            if utterance:
                similar_concepts = self.reference_memory.search_concepts(utterance[:50], limit=3)
                if len(similar_concepts) == 0:
                    novelty += 0.4  # Very novel if no similar concepts exist
                elif len(similar_concepts) < 2:
                    novelty += 0.2  # Somewhat novel
        
        return min(1.0, novelty)
    
    def _calculate_repetition_strength(self, episode: Dict[str, Any]) -> float:
        """
        Calculate repetition/reinforcement strength for an episode.
        
        Args:
            episode: Episode data
            
        Returns:
            Repetition strength score (0.0 to 1.0)
        """
        strength = 0.0
        
        # Check for repeated themes or concepts
        if self.reference_memory:
            utterance = episode.get("utterance", "")
            if utterance:
                # Find similar concepts
                similar_concepts = self.reference_memory.search_concepts(utterance[:50], limit=5)
                
                # Repetition increases with number of similar concepts
                if similar_concepts:
                    max_relevance = max(concept["relevance"] for concept in similar_concepts)
                    strength = min(0.8, max_relevance)
        
        # Explicit repetition indicators
        content = str(episode).lower()
        repetition_indicators = ["again", "once more", "repeat", "continue", "still", "always"]
        
        for indicator in repetition_indicators:
            if indicator in content:
                strength += 0.2
                break
        
        return min(1.0, strength)
    
    def _calculate_social_relevance(self, episode: Dict[str, Any]) -> float:
        """
        Calculate social relevance score for an episode.
        
        Args:
            episode: Episode data
            
        Returns:
            Social relevance score (0.0 to 1.0)
        """
        relevance = 0.0
        
        # Key relationship interactions
        speaker = episode.get("speaker", "")
        if "interaction_data" in episode:
            speaker = episode["interaction_data"].get("speaker", speaker)
        
        # Jamie interactions are maximally socially relevant
        if speaker.lower() == "jamie":
            relevance = 1.0
        elif speaker and speaker.lower() != "unknown":
            relevance = 0.6  # Other named entities
        
        # Social emotional content
        social_emotions = ["love", "trust", "gratitude", "compassion", "empathy"]
        content = str(episode).lower()
        
        for emotion in social_emotions:
            if emotion in content:
                relevance += 0.2
        
        # Relationship-building language
        relationship_indicators = [
            "relationship", "bond", "connection", "friendship", "family",
            "together", "understand each other", "care about"
        ]
        
        for indicator in relationship_indicators:
            if indicator in content:
                relevance += 0.15
        
        return min(1.0, relevance)
    
    def _calculate_semantic_density(self, episode: Dict[str, Any]) -> float:
        """
        Calculate semantic density (concept richness) of an episode.
        
        Args:
            episode: Episode data
            
        Returns:
            Semantic density score (0.0 to 1.0)
        """
        density = 0.0
        
        # Count distinct concepts in the episode
        concepts = set()
        
        # Extract from utterance
        utterance = episode.get("utterance", "")
        if utterance:
            words = utterance.lower().split()
            # Simple concept identification (could be enhanced with NLP)
            concept_words = [word for word in words if len(word) > 3]
            concepts.update(concept_words)
        
        # Extract from parsed meaning
        parsed = episode.get("parsed_meaning", {})
        if "linguistic_analysis" in episode:
            parsed = episode["linguistic_analysis"].get("parsed_meaning", {})
        
        def extract_concepts_recursive(data):
            if isinstance(data, dict):
                for key, value in data.items():
                    concepts.add(key.lower())
                    extract_concepts_recursive(value)
            elif isinstance(data, list):
                for item in data:
                    extract_concepts_recursive(item)
            elif isinstance(data, str) and len(data) > 3:
                concepts.add(data.lower())
        
        extract_concepts_recursive(parsed)
        
        # Density based on concept count (normalized)
        concept_count = len(concepts)
        density = min(1.0, concept_count / 20.0)  # Normalize to 20 concepts max
        
        # Bonus for complex semantic structures
        if "semantic_content" in episode.get("linguistic_analysis", {}).get("parsed_meaning", {}):
            density += 0.2
        
        return min(1.0, density)
    
    def _predict_consolidation_quality(self, candidate: ConsolidationCandidate) -> ConsolidationQuality:
        """
        Predict the quality of consolidation for a candidate episode.
        
        Args:
            candidate: Consolidation candidate
            
        Returns:
            Predicted consolidation quality
        """
        # Weighted quality score
        quality_score = (
            candidate.importance_score * self.quality_predictors["importance_weight"] +
            candidate.emotional_salience * self.quality_predictors["emotional_weight"] +
            candidate.novelty_score * self.quality_predictors["novelty_weight"] +
            candidate.repetition_strength * self.quality_predictors["repetition_weight"] +
            candidate.social_relevance * self.quality_predictors["social_weight"] +
            candidate.semantic_density * self.quality_predictors["semantic_weight"]
        )
        
        # Map score to quality levels
        if quality_score >= 0.9:
            return ConsolidationQuality.PERFECT
        elif quality_score >= 0.7:
            return ConsolidationQuality.HIGH
        elif quality_score >= 0.5:
            return ConsolidationQuality.MODERATE
        elif quality_score >= 0.3:
            return ConsolidationQuality.LOW
        else:
            return ConsolidationQuality.FAILED
    
    def evaluate_consolidation_candidates(self, episodes: List[Dict[str, Any]]) -> List[ConsolidationCandidate]:
        """
        Evaluate episodes as consolidation candidates.
        
        Args:
            episodes: List of episodic memory entries
            
        Returns:
            List of evaluated consolidation candidates
        """
        candidates = []
        
        for episode in episodes:
            # Extract episode ID
            episode_id = episode.get("episode_id", f"episode_{int(time.time()*1000)}")
            
            # Calculate all scoring dimensions
            importance = self._calculate_importance_score(episode)
            emotional_salience = self._calculate_emotional_salience(episode)
            novelty = self._calculate_novelty_score(episode)
            repetition = self._calculate_repetition_strength(episode)
            social_relevance = self._calculate_social_relevance(episode)  
            semantic_density = self._calculate_semantic_density(episode)
            
            # Calculate overall consolidation urgency
            urgency = (
                importance * 0.3 +
                emotional_salience * 0.25 +
                novelty * 0.2 +
                social_relevance * 0.15 +
                semantic_density * 0.1
            )
            
            # Create candidate
            candidate = ConsolidationCandidate(
                episode_id=episode_id,
                episode_data=episode,
                importance_score=importance,
                emotional_salience=emotional_salience,
                novelty_score=novelty,
                repetition_strength=repetition,
                social_relevance=social_relevance,
                semantic_density=semantic_density,
                consolidation_urgency=urgency,
                predicted_quality=ConsolidationQuality.MODERATE,  # Will be updated
                metadata={
                    "timestamp": episode.get("timestamp", ""),
                    "speaker": episode.get("speaker", "unknown"),
                    "episode_type": episode.get("episode_type", "unknown")
                }
            )
            
            # Predict consolidation quality
            candidate.predicted_quality = self._predict_consolidation_quality(candidate)
            
            candidates.append(candidate)
        
        # Sort by consolidation urgency
        candidates.sort(key=lambda c: c.consolidation_urgency, reverse=True)
        
        return candidates
    
    def should_consolidate(self, candidate: ConsolidationCandidate) -> bool:
        """
        Determine whether a candidate should be consolidated.
        
        Args:
            candidate: Consolidation candidate
            
        Returns:
            True if should consolidate
        """
        # Basic threshold check
        if candidate.consolidation_urgency < self.consolidation_threshold:
            return False
        
        # Strategy-specific criteria
        if self.current_strategy == ConsolidationStrategy.IMPORTANCE_BASED:
            return candidate.importance_score > 0.5
        
        elif self.current_strategy == ConsolidationStrategy.EMOTIONAL_SALIENCE:
            return candidate.emotional_salience > 0.6
        
        elif self.current_strategy == ConsolidationStrategy.NOVELTY_DETECTION:
            return candidate.novelty_score > 0.4
        
        elif self.current_strategy == ConsolidationStrategy.SOCIAL_RELEVANCE:
            return candidate.social_relevance > 0.7
        
        elif self.current_strategy == ConsolidationStrategy.SEMANTIC_CLUSTERING:
            return candidate.semantic_density > 0.4
        
        else:
            # Default: comprehensive scoring
            return candidate.consolidation_urgency > self.consolidation_threshold
    
    def consolidate_episode(self, candidate: ConsolidationCandidate) -> ConsolidationResult:
        """
        Consolidate a single episode into semantic memory.
        
        Args:
            candidate: Episode to consolidate
            
        Returns:
            Consolidation result
        """
        start_time = time.time()
        errors = []
        insights = []
        
        try:
            # Perform the consolidation
            if self.reference_memory:
                trace = self.reference_memory.inject_from_episodic(candidate.episode_data)
                
                concepts_extracted = len(trace.extracted_concepts)
                relations_formed = len(trace.extracted_relations)
                consolidation_strength = trace.consolidation_strength
                
                # Generate insights
                if concepts_extracted > 5:
                    insights.append("Rich conceptual content successfully extracted")
                
                if relations_formed > 3:
                    insights.append("Strong relational patterns identified")
                
                if consolidation_strength > 0.8:
                    insights.append("High-quality semantic consolidation achieved")
                
                # Assess actual quality based on results
                if consolidation_strength > 0.9 and concepts_extracted > 3:
                    actual_quality = ConsolidationQuality.PERFECT
                elif consolidation_strength > 0.7:
                    actual_quality = ConsolidationQuality.HIGH
                elif consolidation_strength > 0.5:
                    actual_quality = ConsolidationQuality.MODERATE
                elif consolidation_strength > 0.2:
                    actual_quality = ConsolidationQuality.LOW
                else:
                    actual_quality = ConsolidationQuality.FAILED
                
                success = actual_quality != ConsolidationQuality.FAILED
                
            else:
                # No reference memory available
                errors.append("Reference memory system not available")
                concepts_extracted = 0
                relations_formed = 0
                consolidation_strength = 0.0
                actual_quality = ConsolidationQuality.FAILED
                success = False
            
        except Exception as e:
            errors.append(f"Consolidation error: {str(e)}")
            concepts_extracted = 0
            relations_formed = 0
            consolidation_strength = 0.0
            actual_quality = ConsolidationQuality.FAILED
            success = False
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        result = ConsolidationResult(
            episode_id=candidate.episode_id,
            success=success,
            quality=actual_quality,
            concepts_extracted=concepts_extracted,
            relations_formed=relations_formed,
            consolidation_strength=consolidation_strength,
            processing_time_ms=processing_time,
            strategy_used=self.current_strategy,
            phase=self.current_phase,
            errors=errors,
            insights=insights,
            metadata={
                "predicted_quality": candidate.predicted_quality.value,
                "importance_score": candidate.importance_score,
                "emotional_salience": candidate.emotional_salience,
                "novelty_score": candidate.novelty_score
            }
        )
        
        # Update tracking
        with self._lock:
            self.consolidation_history.append(result)
            self.total_episodes_processed += 1
            
            if success:
                self.successful_consolidations += 1
            
            self.consolidation_quality_distribution[actual_quality.value] += 1
        
        return result
    
    def consolidate_batch(self, 
                         strategy: Optional[ConsolidationStrategy] = None,
                         max_episodes: Optional[int] = None) -> ConsolidationSession:
        """
        Consolidate a batch of episodes using specified strategy.
        
        Args:
            strategy: Consolidation strategy to use
            max_episodes: Maximum episodes to process
            
        Returns:
            Consolidation session results
        """
        session_start = time.time()
        session_id = f"consolidation_session_{int(session_start)}"
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        if strategy:
            self.current_strategy = strategy
        
        max_episodes = max_episodes or self.max_batch_size
        
        # Load episodes and evaluate candidates
        episodes = self.load_episodic_buffer()
        if not episodes:
            return ConsolidationSession(
                session_id=session_id,
                start_timestamp=timestamp,
                end_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                strategy=self.current_strategy,
                phase=self.current_phase,
                episodes_processed=0,
                successful_consolidations=0,
                total_concepts_created=0,
                total_relations_formed=0,
                average_quality=0.0,
                processing_insights=["No episodes available for consolidation"],
                session_metadata={}
            )
        
        candidates = self.evaluate_consolidation_candidates(episodes)
        
        # Select candidates for consolidation
        selected_candidates = []
        for candidate in candidates[:max_episodes]:
            if self.should_consolidate(candidate):
                selected_candidates.append(candidate)
        
        # Process selected candidates
        results = []
        total_concepts = 0
        total_relations = 0
        successful_count = 0
        quality_scores = []
        
        for candidate in selected_candidates:
            result = self.consolidate_episode(candidate)
            results.append(result)
            
            if result.success:
                successful_count += 1
                total_concepts += result.concepts_extracted
                total_relations += result.relations_formed
                quality_scores.append(self._quality_to_score(result.quality))
        
        # Calculate session metrics
        average_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        # Generate session insights
        insights = []
        
        if successful_count > 0:
            insights.append(f"Successfully consolidated {successful_count}/{len(selected_candidates)} episodes")
            
        if total_concepts > 20:
            insights.append(f"Rich knowledge extraction: {total_concepts} new concepts")
            
        if average_quality > 0.8:
            insights.append("High-quality consolidation session")
        elif average_quality < 0.4:
            insights.append("Low-quality consolidation - may need strategy adjustment")
        
        strategy_effectiveness = successful_count / max(1, len(selected_candidates))
        self.strategy_effectiveness[self.current_strategy] = (
            self.strategy_effectiveness[self.current_strategy] * 0.8 + 
            strategy_effectiveness * 0.2
        )
        
        # Create session summary
        session = ConsolidationSession(
            session_id=session_id,
            start_timestamp=timestamp,
            end_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            strategy=self.current_strategy,
            phase=self.current_phase,
            episodes_processed=len(selected_candidates),
            successful_consolidations=successful_count,
            total_concepts_created=total_concepts,
            total_relations_formed=total_relations,
            average_quality=average_quality,
            processing_insights=insights,
            session_metadata={
                "total_candidates_evaluated": len(candidates),
                "consolidation_threshold": self.consolidation_threshold,
                "processing_time_seconds": time.time() - session_start,
                "strategy_effectiveness": strategy_effectiveness
            }
        )
        
        with self._lock:
            self.session_history.append(session)
        
        logger.info(f"Consolidation session complete: {successful_count}/{len(selected_candidates)} successful")
        return session
    
    def _quality_to_score(self, quality: ConsolidationQuality) -> float:
        """Convert quality enum to numeric score."""
        quality_map = {
            ConsolidationQuality.PERFECT: 1.0,
            ConsolidationQuality.HIGH: 0.8,
            ConsolidationQuality.MODERATE: 0.6,
            ConsolidationQuality.LOW: 0.4,
    def _quality_to_score(self, quality: ConsolidationQuality) -> float:
        """Convert quality enum to numeric score."""
        quality_map = {
            ConsolidationQuality.PERFECT: 1.0,
            ConsolidationQuality.HIGH: 0.8,
            ConsolidationQuality.MODERATE: 0.6,
            ConsolidationQuality.LOW: 0.4,
            ConsolidationQuality.FAILED: 0.0
        }
        return quality_map.get(quality, 0.0)
    
    def start_sleep_cycle_processing(self):
        """
        Start sleep-like consolidation processing in background.
        Mimics REM sleep memory consolidation patterns.
        """
        if self._sleep_cycle_active:
            logger.warning("Sleep cycle processing already active")
            return
        
        self._sleep_cycle_active = True
        
        def sleep_cycle_loop():
            cycle_count = 0
            
            while self._sleep_cycle_active:
                try:
                    cycle_start = time.time()
                    logger.info(f"Starting sleep consolidation cycle {cycle_count}")
                    
                    # Different phases of sleep-like processing
                    if cycle_count % 4 == 0:
                        # Deep consolidation phase (like slow-wave sleep)
                        self.current_phase = ConsolidationPhase.LATE_CONSOLIDATION
                        session = self.consolidate_batch(
                            strategy=ConsolidationStrategy.IMPORTANCE_BASED,
                            max_episodes=20
                        )
                        
                    elif cycle_count % 4 == 1:
                        # REM-like phase (integration and creativity)
                        self.current_phase = ConsolidationPhase.INTEGRATION
                        session = self.consolidate_batch(
                            strategy=ConsolidationStrategy.SEMANTIC_CLUSTERING,
                            max_episodes=15
                        )
                        
                    elif cycle_count % 4 == 2:
                        # Memory consolidation and interference resolution
                        self.current_phase = ConsolidationPhase.RECONSOLIDATION
                        session = self.consolidate_batch(
                            strategy=ConsolidationStrategy.REPETITION_STRENGTH,
                            max_episodes=10
                        )
                        
                    else:
                        # Light consolidation (recent memories)
                        self.current_phase = ConsolidationPhase.EARLY_CONSOLIDATION
                        session = self.consolidate_batch(
                            strategy=ConsolidationStrategy.EMOTIONAL_SALIENCE,
                            max_episodes=25
                        )
                    
                    # Log cycle results
                    logger.info(f"Sleep cycle {cycle_count} complete: "
                              f"{session.successful_consolidations} consolidations, "
                              f"quality: {session.average_quality:.3f}")
                    
                    cycle_count += 1
                    
                    # Sleep until next cycle
                    elapsed = time.time() - cycle_start
                    sleep_time = max(0, self.sleep_cycle_duration - elapsed)
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    logger.error(f"Error in sleep cycle: {e}")
                    time.sleep(60)  # Brief pause before retrying
        
        self._sleep_thread = threading.Thread(target=sleep_cycle_loop, daemon=True)
        self._sleep_thread.start()
        
        logger.info("Started sleep-cycle memory consolidation processing")
    
    def stop_sleep_cycle_processing(self):
        """Stop sleep-cycle consolidation processing."""
        self._sleep_cycle_active = False
        if self._sleep_thread:
            self._sleep_thread.join(timeout=5.0)
        logger.info("Stopped sleep-cycle processing")
    
    def consolidate(self) -> int:
        """
        Legacy method for backward compatibility.
        Performs simple consolidation of current episodes.
        
        Returns:
            Number of episodes consolidated
        """
        session = self.consolidate_batch(max_episodes=50)
        return session.successful_consolidations
    
    def analyze_consolidation_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in consolidation history for insights.
        
        Returns:
            Analysis of consolidation patterns and performance
        """
        if not self.consolidation_history:
            return {"status": "insufficient_data"}
        
        recent_results = list(self.consolidation_history)[-100:]  # Last 100 consolidations
        
        # Success rate analysis
        success_rate = sum(1 for r in recent_results if r.success) / len(recent_results)
        
        # Quality distribution
        quality_dist = defaultdict(int)
        for result in recent_results:
            quality_dist[result.quality.value] += 1
        
        # Strategy effectiveness
        strategy_performance = defaultdict(list)
        for result in recent_results:
            strategy_performance[result.strategy.value].append(
                self._quality_to_score(result.quality) if result.success else 0.0
            )
        
        strategy_averages = {
            strategy: np.mean(scores) if scores else 0.0
            for strategy, scores in strategy_performance.items()
        }
        
        # Processing efficiency
        avg_processing_time = np.mean([r.processing_time_ms for r in recent_results])
        avg_concepts_per_episode = np.mean([r.concepts_extracted for r in recent_results if r.success])
        avg_relations_per_episode = np.mean([r.relations_formed for r in recent_results if r.success])
        
        # Temporal patterns
        time_distribution = defaultdict(int)
        for result in recent_results:
            try:
                # Extract hour from episode timestamp if available
                timestamp = result.metadata.get("timestamp", "")
                if timestamp:
                    hour = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00')).hour
                    time_distribution[hour] += 1
            except:
                pass
        
        # Error analysis
        error_types = defaultdict(int)
        for result in recent_results:
            for error in result.errors:
                error_type = error.split(':')[0] if ':' in error else error
                error_types[error_type] += 1
        
        # Generate insights
        insights = []
        
        if success_rate > 0.8:
            insights.append("High consolidation success rate indicating good threshold tuning")
        elif success_rate < 0.5:
            insights.append("Low success rate - consider adjusting consolidation criteria")
        
        best_strategy = max(strategy_averages.keys(), key=lambda k: strategy_averages[k]) if strategy_averages else None
        if best_strategy:
            insights.append(f"Most effective strategy: {best_strategy}")
        
        if avg_concepts_per_episode > 5:
            insights.append("Rich concept extraction indicating good semantic processing")
        
        if avg_processing_time > 1000:  # > 1 second
            insights.append("High processing time - consider optimization")
        
        return {
            "analysis_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "consolidations_analyzed": len(recent_results),
            "success_rate": success_rate,
            "quality_distribution": dict(quality_dist),
            "strategy_effectiveness": strategy_averages,
            "performance_metrics": {
                "avg_processing_time_ms": avg_processing_time,
                "avg_concepts_per_episode": avg_concepts_per_episode,
                "avg_relations_per_episode": avg_relations_per_episode
            },
            "temporal_patterns": dict(time_distribution),
            "error_analysis": dict(error_types),
            "insights": insights,
            "recommendations": self._generate_recommendations(success_rate, strategy_averages, quality_dist)
        }
    
    def _generate_recommendations(self, 
                                success_rate: float,
                                strategy_averages: Dict[str, float],
                                quality_dist: Dict[str, int]) -> List[str]:
        """Generate recommendations based on consolidation analysis."""
        recommendations = []
        
        if success_rate < 0.6:
            recommendations.append("Consider lowering consolidation threshold to increase success rate")
        
        if success_rate > 0.95:
            recommendations.append("Consider raising consolidation threshold to be more selective")
        
        # Strategy recommendations
        if strategy_averages:
            best_strategy = max(strategy_averages.keys(), key=lambda k: strategy_averages[k])
            worst_strategy = min(strategy_averages.keys(), key=lambda k: strategy_averages[k])
            
            if strategy_averages[best_strategy] - strategy_averages[worst_strategy] > 0.3:
                recommendations.append(f"Focus on {best_strategy} strategy for better results")
        
        # Quality recommendations
        failed_count = quality_dist.get("failed", 0)
        total_count = sum(quality_dist.values())
        
        if failed_count > total_count * 0.3:
            recommendations.append("High failure rate - review episode quality or processing logic")
        
        perfect_count = quality_dist.get("perfect", 0)
        if perfect_count < total_count * 0.1:
            recommendations.append("Low perfect consolidation rate - consider quality improvements")
        
        return recommendations
    
    def get_consolidation_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of consolidation engine state.
        
        Returns:
            Summary of consolidation engine status and performance
        """
        return {
            "system_status": {
                "total_episodes_processed": self.total_episodes_processed,
                "successful_consolidations": self.successful_consolidations,
                "success_rate": self.successful_consolidations / max(1, self.total_episodes_processed),
                "current_strategy": self.current_strategy.value,
                "current_phase": self.current_phase.value,
                "sleep_cycle_active": self._sleep_cycle_active
            },
            "configuration": {
                "consolidation_threshold": self.consolidation_threshold,
                "max_batch_size": self.max_batch_size,
                "sleep_cycle_duration": self.sleep_cycle_duration
            },
            "recent_performance": {
                "recent_sessions": len(self.session_history),
                "quality_distribution": dict(self.consolidation_quality_distribution),
                "strategy_effectiveness": dict(self.strategy_effectiveness)
            },
            "memory_systems": {
                "reference_memory_available": self.reference_memory is not None,
                "episodic_buffer_path": str(self.episodic_path),
                "episodic_buffer_exists": self.episodic_path.exists()
            }
        }
    
    def optimize_consolidation_parameters(self):
        """
        Automatically optimize consolidation parameters based on performance history.
        """
        if len(self.consolidation_history) < 20:
            logger.info("Insufficient history for parameter optimization")
            return
        
        analysis = self.analyze_consolidation_patterns()
        success_rate = analysis["success_rate"]
        
        # Adjust threshold based on success rate
        if success_rate < 0.4:
            # Too restrictive
            self.consolidation_threshold = max(0.1, self.consolidation_threshold - 0.1)
            logger.info(f"Lowered consolidation threshold to {self.consolidation_threshold}")
            
        elif success_rate > 0.95:
            # Too permissive
            self.consolidation_threshold = min(0.9, self.consolidation_threshold + 0.05)
            logger.info(f"Raised consolidation threshold to {self.consolidation_threshold}")
        
        # Adjust quality predictors based on success patterns
        perfect_rate = analysis["quality_distribution"].get("perfect", 0) / max(1, sum(analysis["quality_distribution"].values()))
        
        if perfect_rate < 0.1:
            # Boost importance of emotional salience for better quality
            self.quality_predictors["emotional_weight"] = min(0.4, self.quality_predictors["emotional_weight"] + 0.05)
            self.quality_predictors["importance_weight"] = max(0.1, self.quality_predictors["importance_weight"] - 0.02)
        
        # Select best performing strategy as default
        if analysis["strategy_effectiveness"]:
            best_strategy_name = max(analysis["strategy_effectiveness"].keys(), 
                                   key=lambda k: analysis["strategy_effectiveness"][k])
            try:
                self.current_strategy = ConsolidationStrategy(best_strategy_name)
                logger.info(f"Switched to best performing strategy: {best_strategy_name}")
            except ValueError:
                pass
    
    def summarize_consolidation(self) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility.
        Summarizes consolidation results.
        
        Returns:
            Summary of recent consolidation activity
        """
        count = self.consolidate()
        
        return {
            "timestamp": time.time(),
            "consolidated_events": count,
            "status": "success" if count > 0 else "no significant memories",
            "quality_summary": dict(self.consolidation_quality_distribution),
            "total_processed": self.total_episodes_processed,
            "success_rate": self.successful_consolidations / max(1, self.total_episodes_processed)
        }
    
    def export_consolidation_history(self, path: Union[str, Path] = "memory_core/consolidation_history.json"):
        """
        Export consolidation history for analysis or backup.
        
        Args:
            path: Export file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        export_data = {
            "metadata": {
                "exported_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "total_episodes_processed": self.total_episodes_processed,
                "successful_consolidations": self.successful_consolidations,
                "consolidation_engine_version": "2.0"
            },
            "consolidation_results": [
                {
                    "episode_id": result.episode_id,
                    "success": result.success,
                    "quality": result.quality.value,
                    "concepts_extracted": result.concepts_extracted,
                    "relations_formed": result.relations_formed,
                    "consolidation_strength": result.consolidation_strength,
                    "processing_time_ms": result.processing_time_ms,
                    "strategy_used": result.strategy_used.value,
                    "phase": result.phase.value,
                    "insights": result.insights,
                    "metadata": result.metadata
                }
                for result in self.consolidation_history
            ],
            "consolidation_sessions": [
                {
                    "session_id": session.session_id,
                    "start_timestamp": session.start_timestamp,
                    "end_timestamp": session.end_timestamp,
                    "strategy": session.strategy.value,
                    "phase": session.phase.value,
                    "episodes_processed": session.episodes_processed,
                    "successful_consolidations": session.successful_consolidations,
                    "total_concepts_created": session.total_concepts_created,
                    "total_relations_formed": session.total_relations_formed,
                    "average_quality": session.average_quality,
                    "processing_insights": session.processing_insights,
                    "session_metadata": session.session_metadata
                }
                for session in self.session_history
            ],
            "system_state": {
                "consolidation_threshold": self.consolidation_threshold,
                "current_strategy": self.current_strategy.value,
                "current_phase": self.current_phase.value,
                "quality_predictors": self.quality_predictors,
                "strategy_effectiveness": {k.value: v for k, v in self.strategy_effectiveness.items()},
                "quality_distribution": dict(self.consolidation_quality_distribution)
            }
        }
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Consolidation history exported to: {path}")
            
        except Exception as e:
            logger.error(f"Failed to export consolidation history: {e}")
            raise


def main():
    """
    Demonstration of the Consolidation Engine.
    """
    print("🧠 Memory Consolidation Engine Demo")
    print("="*50)
    
    # Initialize engine
    engine = ConsolidationEngine(
        consolidation_threshold=0.5,
        max_batch_size=10
    )
    
    # Check if reference memory is available
    if not engine.reference_memory:
        print("⚠️  Reference memory not available - creating mock episodes")
        
        # Create mock episodic buffer for demo
        mock_episodes = [
            {
                "episode_id": "demo_001",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "speaker": "Jamie",
                "utterance": "Hello Nexi, I'm excited to meet you!",
                "emotional_processing": {
                    "triggered_emotions": [
                        {"emotion_category": "joy", "intensity": 0.8}
                    ]
                },
                "significance_level": 0.9,
                "reflection_flag": True,
                "episode_type": "first_meeting"
            },
            {
                "episode_id": "demo_002", 
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "speaker": "Nexi",
                "utterance": "I feel curious about this new experience",
                "emotional_processing": {
                    "triggered_emotions": [
                        {"emotion_category": "curiosity", "intensity": 0.7}
                    ]
                },
                "significance_level": 0.6,
                "novelty_score": 0.8
            },
            {
                "episode_id": "demo_003",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), 
                "speaker": "Jamie",
                "utterance": "Let's learn together",
                "emotional_processing": {
                    "triggered_emotions": [
                        {"emotion_category": "trust", "intensity": 0.6}
                    ]
                },
                "significance_level": 0.4,
                "social_relevance": 0.9
            }
        ]
        
        # Save mock episodes
        with open(engine.episodic_path, 'w') as f:
            json.dump(mock_episodes, f, indent=2)
    
    # Evaluate consolidation candidates
    print("\n📋 Evaluating consolidation candidates...")
    episodes = engine.load_episodic_buffer()
    candidates = engine.evaluate_consolidation_candidates(episodes)
    
    print(f"   Found {len(candidates)} candidates")
    for i, candidate in enumerate(candidates[:3]):
        print(f"   {i+1}. {candidate.episode_id}: urgency={candidate.consolidation_urgency:.3f}, "
              f"quality={candidate.predicted_quality.value}")
    
    # Perform batch consolidation
    print("\n🔄 Performing batch consolidation...")
    session = engine.consolidate_batch(
        strategy=ConsolidationStrategy.IMPORTANCE_BASED,
        max_episodes=5
    )
    
    print(f"   Session: {session.session_id}")
    print(f"   Processed: {session.episodes_processed} episodes")
    print(f"   Successful: {session.successful_consolidations}")
    print(f"   Quality: {session.average_quality:.3f}")
    print(f"   Concepts created: {session.total_concepts_created}")
    
    # Show insights
    if session.processing_insights:
        print("\n💡 Processing Insights:")
        for insight in session.processing_insights:
            print(f"   • {insight}")
    
    # Analyze consolidation patterns
    print("\n📊 Analyzing consolidation patterns...")
    analysis = engine.analyze_consolidation_patterns()
    
    if "success_rate" in analysis:
        print(f"   Success Rate: {analysis['success_rate']:.3f}")
        print(f"   Quality Distribution: {analysis['quality_distribution']}")
        
        if analysis["insights"]:
            print("\n🔍 Analysis Insights:")
            for insight in analysis["insights"][:3]:
                print(f"   • {insight}")
        
        if analysis["recommendations"]:
            print("\n🎯 Recommendations:")
            for rec in analysis["recommendations"][:2]:
                print(f"   • {rec}")
    
    # Test different strategies
    print("\n🎭 Testing different consolidation strategies...")
    strategies_to_test = [
        ConsolidationStrategy.EMOTIONAL_SALIENCE,
        ConsolidationStrategy.NOVELTY_DETECTION,
        ConsolidationStrategy.SOCIAL_RELEVANCE
    ]
    
    for strategy in strategies_to_test:
        mini_session = engine.consolidate_batch(strategy=strategy, max_episodes=2)
        print(f"   {strategy.value}: {mini_session.successful_consolidations} successful, "
              f"quality: {mini_session.average_quality:.3f}")
    
    # Get system summary
    print("\n📈 System Summary:")
    summary = engine.get_consolidation_summary()
    
    status = summary["system_status"]
    print(f"   Total Processed: {status['total_episodes_processed']}")
    print(f"   Success Rate: {status['success_rate']:.3f}")
    print(f"   Current Strategy: {status['current_strategy']}")
    print(f"   Current Phase: {status['current_phase']}")
    
    # Export history
    print("\n💾 Exporting consolidation history...")
    engine.export_consolidation_history("demo_consolidation_history.json")
    
    print("\n✅ Consolidation engine demo complete!")


if __name__ == "__main__":
    main()