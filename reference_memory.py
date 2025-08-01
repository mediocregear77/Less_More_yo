"""
Reference Memory System

Implements semantic memory and knowledge graph management for Nexi's consciousness.
Provides concept encoding, relationship mapping, knowledge retrieval, and semantic
analysis capabilities for building and maintaining a rich conceptual understanding.

Core Functions:
- Semantic concept encoding and categorization
- Multi-dimensional relationship mapping
- Knowledge graph construction and maintenance
- Episodic-to-semantic memory consolidation
- Concept clustering and similarity analysis
- Temporal knowledge evolution tracking
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
import networkx as nx
import hashlib
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConceptType(Enum):
    """Types of concepts in the knowledge graph."""
    ENTITY = "entity"                    # People, objects, places
    EMOTION = "emotion"                  # Emotional states and feelings
    ACTION = "action"                    # Behaviors and activities
    PROPERTY = "property"                # Attributes and characteristics
    RELATIONSHIP = "relationship"        # Social and logical connections
    EVENT = "event"                      # Temporal occurrences
    ABSTRACT = "abstract"                # Ideas and concepts
    GOAL = "goal"                        # Intentions and objectives
    BELIEF = "belief"                    # Convictions and opinions
    MEMORY = "memory"                    # Remembered experiences
    SKILL = "skill"                      # Learned capabilities
    PATTERN = "pattern"                  # Recognized regularities

class RelationType(Enum):
    """Types of relationships between concepts."""
    IS_A = "is_a"                        # Taxonomic relationship
    PART_OF = "part_of"                  # Compositional relationship
    CAUSES = "causes"                    # Causal relationship
    ENABLES = "enables"                  # Enablement relationship
    SIMILAR_TO = "similar_to"            # Similarity relationship
    OPPOSITE_OF = "opposite_of"          # Antonym relationship
    TEMPORAL_BEFORE = "temporal_before"  # Temporal precedence
    TEMPORAL_AFTER = "temporal_after"    # Temporal succession
    ASSOCIATED_WITH = "associated_with"  # General association
    USED_FOR = "used_for"               # Functional relationship
    LOCATED_IN = "located_in"           # Spatial relationship
    OWNS = "owns"                       # Possession relationship
    LOVES = "loves"                     # Emotional attachment
    FEARS = "fears"                     # Emotional aversion
    REMEMBERS = "remembers"             # Memory relationship
    LEARNS_FROM = "learns_from"         # Learning relationship
    INTERACTS_WITH = "interacts_with"   # Social interaction
    CREATES = "creates"                 # Creation relationship
    EXPERIENCES = "experiences"         # Experiential relationship

@dataclass
class ConceptNode:
    """Rich concept representation in the knowledge graph."""
    concept_id: str
    name: str
    concept_type: ConceptType
    definition: Optional[str]
    aliases: List[str]
    properties: Dict[str, Any]
    activation_level: float
    importance_score: float
    creation_timestamp: str
    last_accessed: str
    access_count: int
    emotional_valence: float
    confidence_score: float
    source_episodes: List[str]
    metadata: Dict[str, Any]

@dataclass
class ConceptRelation:
    """Rich relationship representation between concepts."""
    source_concept: str
    target_concept: str
    relation_type: RelationType
    strength: float
    confidence: float
    temporal_weight: float
    creation_timestamp: str
    last_reinforced: str
    reinforcement_count: int
    source_episodes: List[str]
    bidirectional: bool
    metadata: Dict[str, Any]

@dataclass
class MemoryTrace:
    """Trace from episodic to semantic memory consolidation."""
    trace_id: str
    timestamp: str
    episode_id: Optional[str]
    extracted_concepts: List[str]
    extracted_relations: List[Tuple[str, str, str]]
    consolidation_strength: float
    processing_notes: List[str]
    metadata: Dict[str, Any]

class ReferenceMemory:
    """
    Advanced semantic memory system implementing knowledge graph representation,
    concept learning, and episodic-to-semantic memory consolidation.
    """
    
    def __init__(self, 
                 memory_path: Union[str, Path] = "memory_core/reference_memory_graph.json",
                 concept_threshold: float = 0.1,
                 decay_rate: float = 0.01,
                 max_concepts: int = 50000):
        """
        Initialize the reference memory system.
        
        Args:
            memory_path: Path to persistent memory storage
            concept_threshold: Minimum activation for concept retention
            decay_rate: Rate of concept activation decay
            max_concepts: Maximum concepts to maintain
        """
        self.memory_path = Path(memory_path)
        self.concept_threshold = concept_threshold
        self.decay_rate = decay_rate
        self.max_concepts = max_concepts
        
        # Core knowledge graph
        self.knowledge_graph = nx.MultiDiGraph()
        self.concepts: Dict[str, ConceptNode] = {}
        self.relations: Dict[str, ConceptRelation] = {}
        
        # Processing components
        self.memory_traces = deque(maxlen=10000)
        self.concept_clusters: Dict[str, Set[str]] = defaultdict(set)
        self.activation_history = defaultdict(list)
        
        # System state
        self.total_concepts = 0
        self.total_relations = 0
        self.consolidation_count = 0
        self.last_cleanup_time = None
        
        # Threading for background processing
        self._processing_active = False
        self._processing_thread = None
        self._lock = threading.Lock()
        
        # Load existing memory
        self.load_memory()
        
        logger.info(f"Reference Memory initialized with {self.total_concepts} concepts")
    
    def _generate_concept_id(self, name: str) -> str:
        """Generate unique concept identifier."""
        # Create deterministic ID based on normalized name
        normalized = re.sub(r'[^\w\s]', '', name.lower().strip())
        normalized = re.sub(r'\s+', '_', normalized)
        hash_suffix = hashlib.md5(normalized.encode()).hexdigest()[:8]
        return f"concept_{normalized}_{hash_suffix}"
    
    def _generate_relation_id(self, source: str, target: str, relation_type: str) -> str:
        """Generate unique relation identifier."""
        relation_str = f"{source}_{relation_type}_{target}"
        hash_suffix = hashlib.md5(relation_str.encode()).hexdigest()[:8]
        return f"rel_{hash_suffix}"
    
    def _normalize_concept_name(self, name: str) -> str:
        """Normalize concept name for consistency."""
        # Remove extra whitespace and convert to title case
        normalized = ' '.join(name.strip().split())
        return normalized.lower()
    
    def _extract_concepts_from_text(self, text: str) -> List[Tuple[str, ConceptType]]:
        """
        Extract potential concepts from text using NLP techniques.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of (concept_name, concept_type) tuples
        """
        concepts = []
        text_lower = text.lower()
        
        # Simple pattern-based extraction (could be enhanced with NLP libraries)
        
        # Emotion patterns
        emotion_patterns = [
            r'\b(joy|happiness|sadness|anger|fear|surprise|disgust|love|hate|curiosity|wonder|excitement|anxiety|calm|peace)\b',
            r'\bfeel(?:ing)?\s+(\w+)',
            r'\bem(?:otion|otional)\w*\s+(\w+)'
        ]
        
        for pattern in emotion_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                emotion = match.group(1) if match.groups() else match.group(0)
                concepts.append((emotion, ConceptType.EMOTION))
        
        # Action patterns
        action_patterns = [
            r'\b(thinking|learning|understanding|remembering|forgetting|creating|building|talking|listening|observing|reflecting)\b',
            r'\b(\w+ing)\b'  # General -ing words
        ]
        
        for pattern in action_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                action = match.group(1) if match.groups() else match.group(0)
                if len(action) > 3:  # Filter very short words
                    concepts.append((action, ConceptType.ACTION))
        
        # Entity patterns (proper nouns, names)
        entity_patterns = [
            r'\b[A-Z][a-z]+\b',  # Capitalized words
            r'\b(Jamie|Nexi)\b'   # Specific important entities
        ]
        
        for pattern in entity_patterns:
            matches = re.finditer(pattern, text)  # Use original case
            for match in matches:
                entity = match.group(0)
                concepts.append((entity, ConceptType.ENTITY))
        
        # Abstract concept patterns
        abstract_patterns = [
            r'\b(consciousness|awareness|identity|existence|reality|truth|knowledge|wisdom|intelligence|creativity)\b',
            r'\b(relationship|connection|bond|friendship|family|trust|love|care)\b'
        ]
        
        for pattern in abstract_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                concept = match.group(0)
                concepts.append((concept, ConceptType.ABSTRACT))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_concepts = []
        for concept, ctype in concepts:
            normalized = self._normalize_concept_name(concept)
            if normalized not in seen and len(normalized) > 1:
                seen.add(normalized)
                unique_concepts.append((normalized, ctype))
        
        return unique_concepts
    
    def encode_concept(self, 
                      name: str,
                      concept_type: Optional[ConceptType] = None,
                      definition: Optional[str] = None,
                      properties: Optional[Dict[str, Any]] = None,
                      emotional_valence: float = 0.0,
                      importance: float = 0.5,
                      source_episode: Optional[str] = None) -> str:
        """
        Encode a concept into the knowledge graph.
        
        Args:
            name: Concept name
            concept_type: Type of concept
            definition: Optional definition
            properties: Additional properties
            emotional_valence: Emotional association (-1 to 1)
            importance: Importance score (0 to 1)
            source_episode: Episode that introduced this concept
            
        Returns:
            Concept ID
        """
        normalized_name = self._normalize_concept_name(name)
        concept_id = self._generate_concept_id(normalized_name)
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        with self._lock:
            if concept_id in self.concepts:
                # Update existing concept
                concept = self.concepts[concept_id]
                concept.activation_level = min(1.0, concept.activation_level + 0.1)
                concept.access_count += 1
                concept.last_accessed = timestamp
                
                # Update properties if provided
                if properties:
                    concept.properties.update(properties)
                
                # Add source episode if provided
                if source_episode and source_episode not in concept.source_episodes:
                    concept.source_episodes.append(source_episode)
                
                # Update importance if higher
                if importance > concept.importance_score:
                    concept.importance_score = importance
                
            else:
                # Create new concept
                concept = ConceptNode(
                    concept_id=concept_id,
                    name=normalized_name,
                    concept_type=concept_type or ConceptType.ABSTRACT,
                    definition=definition,
                    aliases=[],
                    properties=properties or {},
                    activation_level=importance,
                    importance_score=importance,
                    creation_timestamp=timestamp,
                    last_accessed=timestamp,
                    access_count=1,
                    emotional_valence=emotional_valence,
                    confidence_score=0.7,
                    source_episodes=[source_episode] if source_episode else [],
                    metadata={}
                )
                
                self.concepts[concept_id] = concept
                self.knowledge_graph.add_node(concept_id, **asdict(concept))
                self.total_concepts += 1
        
        # Track activation
        self.activation_history[concept_id].append((timestamp, importance))
        
        logger.debug(f"Encoded concept: {normalized_name} ({concept_type})")
        return concept_id
    
    def connect_concepts(self,
                        source_name: str,
                        target_name: str,
                        relation_type: Union[RelationType, str],
                        strength: float = 0.5,
                        confidence: float = 0.7,
                        bidirectional: bool = False,
                        source_episode: Optional[str] = None) -> str:
        """
        Create or strengthen a relationship between concepts.
        
        Args:
            source_name: Source concept name
            target_name: Target concept name
            relation_type: Type of relationship
            strength: Relationship strength (0 to 1)
            confidence: Confidence in relationship (0 to 1)
            bidirectional: Whether relationship works both ways
            source_episode: Episode that established this relationship
            
        Returns:
            Relation ID
        """
        # Ensure concepts exist
        source_id = self.encode_concept(source_name)
        target_id = self.encode_concept(target_name)
        
        # Convert string to enum if needed
        if isinstance(relation_type, str):
            try:
                relation_type = RelationType(relation_type)
            except ValueError:
                logger.warning(f"Unknown relation type: {relation_type}, using ASSOCIATED_WITH")
                relation_type = RelationType.ASSOCIATED_WITH
        
        relation_id = self._generate_relation_id(source_id, target_id, relation_type.value)
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        with self._lock:
            if relation_id in self.relations:
                # Strengthen existing relationship
                relation = self.relations[relation_id]
                relation.strength = min(1.0, relation.strength + strength * 0.1)
                relation.confidence = min(1.0, relation.confidence + confidence * 0.05)
                relation.reinforcement_count += 1
                relation.last_reinforced = timestamp
                
                if source_episode and source_episode not in relation.source_episodes:
                    relation.source_episodes.append(source_episode)
                
            else:
                # Create new relationship
                relation = ConceptRelation(
                    source_concept=source_id,
                    target_concept=target_id,
                    relation_type=relation_type,
                    strength=strength,
                    confidence=confidence,
                    temporal_weight=1.0,
                    creation_timestamp=timestamp,
                    last_reinforced=timestamp,
                    reinforcement_count=1,
                    source_episodes=[source_episode] if source_episode else [],
                    bidirectional=bidirectional,
                    metadata={}
                )
                
                self.relations[relation_id] = relation
                self.knowledge_graph.add_edge(
                    source_id, target_id, 
                    key=relation_id,
                    **asdict(relation)
                )
                self.total_relations += 1
                
                # Add reverse relationship if bidirectional
                if bidirectional:
                    reverse_id = self._generate_relation_id(target_id, source_id, relation_type.value)
                    reverse_relation = ConceptRelation(
                        source_concept=target_id,
                        target_concept=source_id,
                        relation_type=relation_type,
                        strength=strength,
                        confidence=confidence,
                        temporal_weight=1.0,
                        creation_timestamp=timestamp,
                        last_reinforced=timestamp,
                        reinforcement_count=1,
                        source_episodes=[source_episode] if source_episode else [],
                        bidirectional=True,
                        metadata={}
                    )
                    
                    self.relations[reverse_id] = reverse_relation
                    self.knowledge_graph.add_edge(
                        target_id, source_id,
                        key=reverse_id,
                        **asdict(reverse_relation)
                    )
        
        logger.debug(f"Connected: {source_name} --{relation_type.value}--> {target_name}")
        return relation_id
    
    def query_related_concepts(self,
                             concept_name: str,
                             relation_types: Optional[List[RelationType]] = None,
                             max_depth: int = 2,
                             min_strength: float = 0.1,
                             limit: int = 20) -> List[Dict[str, Any]]:
        """
        Query concepts related to a given concept.
        
        Args:
            concept_name: Name of the source concept
            relation_types: Filter by specific relation types
            max_depth: Maximum traversal depth
            min_strength: Minimum relationship strength
            limit: Maximum number of results
            
        Returns:
            List of related concept information
        """
        normalized_name = self._normalize_concept_name(concept_name)
        concept_id = self._generate_concept_id(normalized_name)
        
        if concept_id not in self.concepts:
            logger.warning(f"Concept not found: {concept_name}")
            return []
        
        results = []
        visited = set()
        
        def traverse(current_id: str, depth: int, path: List[str]):
            if depth > max_depth or current_id in visited:
                return
            
            visited.add(current_id)
            
            # Get all outgoing edges
            if current_id in self.knowledge_graph:
                for target_id in self.knowledge_graph[current_id]:
                    for edge_key, edge_data in self.knowledge_graph[current_id][target_id].items():
                        relation = self.relations.get(edge_key)
                        
                        if not relation or relation.strength < min_strength:
                            continue
                        
                        if relation_types and relation.relation_type not in relation_types:
                            continue
                        
                        target_concept = self.concepts.get(target_id)
                        if not target_concept:
                            continue
                        
                        # Calculate relevance score
                        relevance = (
                            relation.strength * 0.4 +
                            relation.confidence * 0.3 +
                            target_concept.activation_level * 0.2 +
                            target_concept.importance_score * 0.1
                        ) / (depth + 1)  # Decay with distance
                        
                        result = {
                            "concept_id": target_id,
                            "concept_name": target_concept.name,
                            "concept_type": target_concept.concept_type.value,
                            "relation_type": relation.relation_type.value,
                            "relation_strength": relation.strength,
                            "confidence": relation.confidence,
                            "relevance_score": relevance,
                            "depth": depth,
                            "path": path + [current_id],
                            "activation_level": target_concept.activation_level,
                            "emotional_valence": target_concept.emotional_valence
                        }
                        
                        results.append(result)
                        
                        # Continue traversal if within depth limit
                        if depth < max_depth:
                            traverse(target_id, depth + 1, path + [current_id])
        
        # Start traversal
        traverse(concept_id, 0, [])
        
        # Sort by relevance and limit results
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:limit]
    
    def inject_from_episodic(self, episode_data: Dict[str, Any]) -> MemoryTrace:
        """
        Extract semantic concepts and relationships from episodic memory.
        
        Args:
            episode_data: Episode data from episodic buffer
            
        Returns:
            Memory trace of the consolidation process
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        trace_id = f"trace_{int(time.time()*1000)}_{self.consolidation_count}"
        
        extracted_concepts = []
        extracted_relations = []
        processing_notes = []
        
        try:
            # Extract episode information
            episode_id = episode_data.get("episode_id", "unknown")
            speaker = episode_data.get("speaker", "unknown")
            utterance = episode_data.get("utterance", "")
            
            # Process different episode formats
            if "interaction_data" in episode_data:
                # New format
                interaction = episode_data["interaction_data"]
                speaker = interaction.get("speaker", speaker)
                utterance = interaction.get("utterance", utterance)
            
            # Extract concepts from utterance
            text_concepts = self._extract_concepts_from_text(utterance)
            
            for concept_name, concept_type in text_concepts:
                concept_id = self.encode_concept(
                    name=concept_name,
                    concept_type=concept_type,
                    source_episode=episode_id
                )
                extracted_concepts.append(concept_id)
                processing_notes.append(f"Extracted {concept_type.value}: {concept_name}")
            
            # Extract concepts from parsed meaning
            parsed_meaning = episode_data.get("parsed_meaning", {})
            if "linguistic_analysis" in episode_data:
                parsed_meaning = episode_data["linguistic_analysis"].get("parsed_meaning", {})
            
            for key, value in parsed_meaning.items():
                if isinstance(value, dict):
                    # Process nested structures
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, str):
                            concept_id = self.encode_concept(
                                name=subvalue,
                                concept_type=ConceptType.ABSTRACT,
                                source_episode=episode_id
                            )
                            extracted_concepts.append(concept_id)
                elif isinstance(value, str):
                    concept_id = self.encode_concept(
                        name=value,
                        concept_type=ConceptType.PROPERTY,
                        source_episode=episode_id
                    )
                    extracted_concepts.append(concept_id)
            
            # Extract emotional concepts
            emotional_data = episode_data.get("emotion", episode_data.get("emotional_processing", {}))
            if emotional_data:
                if "triggered_emotion" in emotional_data:
                    emotion_name = emotional_data["triggered_emotion"]
                elif "triggered_emotions" in emotional_data:
                    emotions = emotional_data["triggered_emotions"]
                    if emotions:
                        emotion_name = emotions[0].get("emotion_category", "unknown")
                else:
                    emotion_name = None
                
                if emotion_name:
                    emotion_id = self.encode_concept(
                        name=emotion_name,
                        concept_type=ConceptType.EMOTION,
                        emotional_valence=emotional_data.get("intensity", 0.5),
                        source_episode=episode_id
                    )
                    extracted_concepts.append(emotion_id)
                    processing_notes.append(f"Extracted emotion: {emotion_name}")
            
            # Create relationships
            if speaker and speaker.lower() != "unknown":
                speaker_id = self.encode_concept(
                    name=speaker,
                    concept_type=ConceptType.ENTITY,
                    source_episode=episode_id
                )
                extracted_concepts.append(speaker_id)
                
                # Connect speaker to concepts they mentioned
                for concept_id in extracted_concepts[-3:]:  # Recent concepts
                    if concept_id != speaker_id:
                        relation_id = self.connect_concepts(
                            speaker, 
                            self.concepts[concept_id].name,
                            RelationType.EXPERIENCES,
                            strength=0.6,
                            source_episode=episode_id
                        )
                        extracted_relations.append((speaker, self.concepts[concept_id].name, "experiences"))
            
            # Connect concepts based on co-occurrence
            if len(extracted_concepts) > 1:
                for i, concept_id_1 in enumerate(extracted_concepts):
                    for concept_id_2 in extracted_concepts[i+1:]:
                        if concept_id_1 != concept_id_2:
                            concept_1 = self.concepts[concept_id_1]
                            concept_2 = self.concepts[concept_id_2]
                            
                            relation_id = self.connect_concepts(
                                concept_1.name,
                                concept_2.name,
                                RelationType.ASSOCIATED_WITH,
                                strength=0.3,
                                source_episode=episode_id
                            )
                            extracted_relations.append((concept_1.name, concept_2.name, "associated_with"))
            
            consolidation_strength = min(1.0, len(extracted_concepts) / 10.0)
            processing_notes.append(f"Consolidation strength: {consolidation_strength:.2f}")
            
        except Exception as e:
            logger.error(f"Error processing episode {episode_id}: {e}")
            processing_notes.append(f"Error: {str(e)}")
            consolidation_strength = 0.0
        
        # Create memory trace
        trace = MemoryTrace(
            trace_id=trace_id,
            timestamp=timestamp,
            episode_id=episode_data.get("episode_id"),
            extracted_concepts=extracted_concepts,
            extracted_relations=extracted_relations,
            consolidation_strength=consolidation_strength,
            processing_notes=processing_notes,
            metadata={
                "source_type": "episodic_injection",
                "concepts_count": len(extracted_concepts),
                "relations_count": len(extracted_relations)
            }
        )
        
        with self._lock:
            self.memory_traces.append(trace)
            self.consolidation_count += 1
        
        logger.info(f"Consolidated episode {episode_id}: {len(extracted_concepts)} concepts, {len(extracted_relations)} relations")
        return trace
    
    def find_concept_clusters(self, min_cluster_size: int = 3) -> Dict[str, Set[str]]:
        """
        Identify clusters of highly connected concepts.
        
        Args:
            min_cluster_size: Minimum concepts per cluster
            
        Returns:
            Dictionary of cluster_name -> concept_set
        """
        # Use community detection algorithms
        try:
            # Convert to undirected graph for community detection
            undirected = self.knowledge_graph.to_undirected()
            
            # Remove low-weight edges
            edges_to_remove = []
            for u, v, data in undirected.edges(data=True):
                if data.get('strength', 0) < 0.3:
                    edges_to_remove.append((u, v))
            
            for edge in edges_to_remove:
                undirected.remove_edge(*edge)
            
            # Find communities using Louvain method (simplified)
            clusters = {}
            cluster_id = 0
            
            # Simple connected components as clusters
            for component in nx.connected_components(undirected):
                if len(component) >= min_cluster_size:
                    cluster_name = f"cluster_{cluster_id}"
                    
                    # Get concept names for this cluster
                    concept_names = set()
                    for concept_id in component:
                        if concept_id in self.concepts:
                            concept_names.add(self.concepts[concept_id].name)
                    
                    if concept_names:
                        clusters[cluster_name] = concept_names
                        cluster_id += 1
            
            self.concept_clusters = clusters
            return clusters
            
        except Exception as e:
            logger.error(f"Error finding concept clusters: {e}")
            return {}
    
    def get_concept_by_name(self, name: str) -> Optional[ConceptNode]:
        """Get concept by name."""
        normalized_name = self._normalize_concept_name(name)
        concept_id = self._generate_concept_id(normalized_name)
        return self.concepts.get(concept_id)
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of current knowledge state.
        
        Returns:
            Dictionary containing knowledge statistics and insights
        """
        with self._lock:
            # Basic statistics
            concept_types = defaultdict(int)
            relation_types = defaultdict(int)
            total_activation = 0.0
            
            for concept in self.concepts.values():
                concept_types[concept.concept_type.value] += 1
                total_activation += concept.activation_level
            
            for relation in self.relations.values():
                relation_types[relation.relation_type.value] += 1
            
            # Most active concepts
            sorted_concepts = sorted(
                self.concepts.values(),
                key=lambda c: c.activation_level * c.importance_score,
                reverse=True
            )
            
            top_concepts = [
                {
                    "name": concept.name,
                    "type": concept.concept_type.value,
                    "activation": concept.activation_level,
                    "importance": concept.importance_score,
                    "access_count": concept.access_count
                }
                for concept in sorted_concepts[:10]
            ]
            
            # Recent consolidations
            recent_traces = list(self.memory_traces)[-5:]
            recent_consolidation_info = [
                {
                    "episode_id": trace.episode_id,
                    "concepts_extracted": len(trace.extracted_concepts),
                    "relations_extracted": len(trace.extracted_relations),
                    "strength": trace.consolidation_strength
                }
                for trace in recent_traces
            ]
            
            return {
                "total_concepts": self.total_concepts,
                "total_relations": self.total_relations,
                "concept_types": dict(concept_types),
                "relation_types": dict(relation_types),
                "average_activation": total_activation / max(1, self.total_concepts),
                "consolidation_count": self.consolidation_count,
                "top_concepts": top_concepts,
                "recent_consolidations": recent_consolidation_info,
                "concept_clusters": len(self.concept_clusters),
                "memory_traces": len(self.memory_traces)
            }
    
    def save_memory(self):
        """Save knowledge graph to persistent storage."""
        try:
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare serializable data
            export_data = {
                "metadata": {
                    "saved_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    def save_memory(self):
        """Save knowledge graph to persistent storage."""
        try:
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare serializable data
            export_data = {
                "metadata": {
                    "saved_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "total_concepts": self.total_concepts,
                    "total_relations": self.total_relations,
                    "consolidation_count": self.consolidation_count,
                    "system_version": "2.0"
                },
                "concepts": {
                    concept_id: {
                        "concept_id": concept.concept_id,
                        "name": concept.name,
                        "concept_type": concept.concept_type.value,
                        "definition": concept.definition,
                        "aliases": concept.aliases,
                        "properties": concept.properties,
                        "activation_level": concept.activation_level,
                        "importance_score": concept.importance_score,
                        "creation_timestamp": concept.creation_timestamp,
                        "last_accessed": concept.last_accessed,
                        "access_count": concept.access_count,
                        "emotional_valence": concept.emotional_valence,
                        "confidence_score": concept.confidence_score,
                        "source_episodes": concept.source_episodes,
                        "metadata": concept.metadata
                    }
                    for concept_id, concept in self.concepts.items()
                },
                "relations": {
                    relation_id: {
                        "source_concept": relation.source_concept,
                        "target_concept": relation.target_concept,
                        "relation_type": relation.relation_type.value,
                        "strength": relation.strength,
                        "confidence": relation.confidence,
                        "temporal_weight": relation.temporal_weight,
                        "creation_timestamp": relation.creation_timestamp,
                        "last_reinforced": relation.last_reinforced,
                        "reinforcement_count": relation.reinforcement_count,
                        "source_episodes": relation.source_episodes,
                        "bidirectional": relation.bidirectional,
                        "metadata": relation.metadata
                    }
                    for relation_id, relation in self.relations.items()
                },
                "memory_traces": [
                    {
                        "trace_id": trace.trace_id,
                        "timestamp": trace.timestamp,
                        "episode_id": trace.episode_id,
                        "extracted_concepts": trace.extracted_concepts,
                        "extracted_relations": trace.extracted_relations,
                        "consolidation_strength": trace.consolidation_strength,
                        "processing_notes": trace.processing_notes,
                        "metadata": trace.metadata
                    }
                    for trace in self.memory_traces
                ],
                "concept_clusters": {
                    cluster_name: list(concepts)
                    for cluster_name, concepts in self.concept_clusters.items()
                },
                "system_parameters": {
                    "concept_threshold": self.concept_threshold,
                    "decay_rate": self.decay_rate,
                    "max_concepts": self.max_concepts
                }
            }
            
            with open(self.memory_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Reference memory saved to: {self.memory_path}")
            
        except Exception as e:
            logger.error(f"Failed to save reference memory: {e}")
            raise
    
    def load_memory(self):
        """Load knowledge graph from persistent storage."""
        if not self.memory_path.exists():
            logger.info("No existing reference memory found, starting fresh")
            return
        
        try:
            with open(self.memory_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Handle both old and new formats
            if "concepts" in import_data and "relations" in import_data:
                # New format
                self._load_new_format(import_data)
            else:
                # Old NetworkX format
                self._load_legacy_format(import_data)
            
            logger.info(f"Reference memory loaded: {self.total_concepts} concepts, {self.total_relations} relations")
            
        except Exception as e:
            logger.error(f"Failed to load reference memory: {e}")
            # Start fresh if loading fails
            self.knowledge_graph = nx.MultiDiGraph()
            self.concepts = {}
            self.relations = {}
    
    def _load_new_format(self, import_data: Dict[str, Any]):
        """Load data in new structured format."""
        # Clear existing data
        self.knowledge_graph = nx.MultiDiGraph()
        self.concepts = {}
        self.relations = {}
        self.memory_traces.clear()
        self.concept_clusters.clear()
        
        # Load concepts
        for concept_id, concept_data in import_data.get("concepts", {}).items():
            concept = ConceptNode(
                concept_id=concept_data["concept_id"],
                name=concept_data["name"],
                concept_type=ConceptType(concept_data["concept_type"]),
                definition=concept_data.get("definition"),
                aliases=concept_data.get("aliases", []),
                properties=concept_data.get("properties", {}),
                activation_level=concept_data.get("activation_level", 0.5),
                importance_score=concept_data.get("importance_score", 0.5),
                creation_timestamp=concept_data.get("creation_timestamp", ""),
                last_accessed=concept_data.get("last_accessed", ""),
                access_count=concept_data.get("access_count", 0),
                emotional_valence=concept_data.get("emotional_valence", 0.0),
                confidence_score=concept_data.get("confidence_score", 0.7),
                source_episodes=concept_data.get("source_episodes", []),
                metadata=concept_data.get("metadata", {})
            )
            
            self.concepts[concept_id] = concept
            self.knowledge_graph.add_node(concept_id, **asdict(concept))
        
        # Load relations
        for relation_id, relation_data in import_data.get("relations", {}).items():
            relation = ConceptRelation(
                source_concept=relation_data["source_concept"],
                target_concept=relation_data["target_concept"],
                relation_type=RelationType(relation_data["relation_type"]),
                strength=relation_data.get("strength", 0.5),
                confidence=relation_data.get("confidence", 0.7),
                temporal_weight=relation_data.get("temporal_weight", 1.0),
                creation_timestamp=relation_data.get("creation_timestamp", ""),
                last_reinforced=relation_data.get("last_reinforced", ""),
                reinforcement_count=relation_data.get("reinforcement_count", 1),
                source_episodes=relation_data.get("source_episodes", []),
                bidirectional=relation_data.get("bidirectional", False),
                metadata=relation_data.get("metadata", {})
            )
            
            self.relations[relation_id] = relation
            
            # Add edge to graph if both concepts exist
            if (relation.source_concept in self.concepts and 
                relation.target_concept in self.concepts):
                self.knowledge_graph.add_edge(
                    relation.source_concept,
                    relation.target_concept,
                    key=relation_id,
                    **asdict(relation)
                )
        
        # Load memory traces
        for trace_data in import_data.get("memory_traces", []):
            trace = MemoryTrace(
                trace_id=trace_data["trace_id"],
                timestamp=trace_data["timestamp"],
                episode_id=trace_data.get("episode_id"),
                extracted_concepts=trace_data.get("extracted_concepts", []),
                extracted_relations=trace_data.get("extracted_relations", []),
                consolidation_strength=trace_data.get("consolidation_strength", 0.0),
                processing_notes=trace_data.get("processing_notes", []),
                metadata=trace_data.get("metadata", {})
            )
            self.memory_traces.append(trace)
        
        # Load concept clusters
        for cluster_name, concepts in import_data.get("concept_clusters", {}).items():
            self.concept_clusters[cluster_name] = set(concepts)
        
        # Update counters
        self.total_concepts = len(self.concepts)
        self.total_relations = len(self.relations)
        self.consolidation_count = import_data.get("metadata", {}).get("consolidation_count", 0)
    
    def _load_legacy_format(self, import_data: Dict[str, Any]):
        """Load data in legacy NetworkX format."""
        logger.info("Loading legacy format reference memory")
        
        # Load as NetworkX graph
        self.knowledge_graph = nx.node_link_graph(import_data)
        
        # Convert to new format
        self.concepts = {}
        self.relations = {}
        
        # Convert nodes to concepts
        for node_id, node_data in self.knowledge_graph.nodes(data=True):
            # Try to infer concept type from metadata or default to ABSTRACT
            concept_type = ConceptType.ABSTRACT
            if 'metadata' in node_data and 'type' in node_data['metadata']:
                try:
                    concept_type = ConceptType(node_data['metadata']['type'])
                except ValueError:
                    pass
            
            concept = ConceptNode(
                concept_id=node_id,
                name=node_id,  # Use node_id as name for legacy data
                concept_type=concept_type,
                definition=None,
                aliases=[],
                properties=node_data.get('metadata', {}),
                activation_level=node_data.get('activation', 0.5),
                importance_score=0.5,
                creation_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                last_accessed=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                access_count=1,
                emotional_valence=0.0,
                confidence_score=0.7,
                source_episodes=[],
                metadata=node_data.get('metadata', {})
            )
            
            self.concepts[node_id] = concept
        
        # Convert edges to relations
        relation_counter = 0
        for source, target, edge_data in self.knowledge_graph.edges(data=True):
            relation_type = RelationType.ASSOCIATED_WITH
            if 'relation' in edge_data:
                try:
                    relation_type = RelationType(edge_data['relation'])
                except ValueError:
                    pass
            
            relation_id = f"legacy_rel_{relation_counter}"
            relation = ConceptRelation(
                source_concept=source,
                target_concept=target,
                relation_type=relation_type,
                strength=edge_data.get('weight', 0.5),
                confidence=0.7,
                temporal_weight=1.0,
                creation_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                last_reinforced=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                reinforcement_count=1,
                source_episodes=[],
                bidirectional=False,
                metadata={}
            )
            
            self.relations[relation_id] = relation
            relation_counter += 1
        
        self.total_concepts = len(self.concepts)
        self.total_relations = len(self.relations)
    
    def decay_activation(self, decay_factor: Optional[float] = None):
        """
        Apply natural decay to concept activation levels.
        
        Args:
            decay_factor: Override default decay rate
        """
        if decay_factor is None:
            decay_factor = self.decay_rate
        
        with self._lock:
            concepts_to_remove = []
            
            for concept_id, concept in self.concepts.items():
                # Apply decay
                concept.activation_level *= (1.0 - decay_factor)
                
                # Remove concepts below threshold
                if (concept.activation_level < self.concept_threshold and 
                    concept.importance_score < 0.3):
                    concepts_to_remove.append(concept_id)
            
            # Remove low-activation concepts
            for concept_id in concepts_to_remove:
                self._remove_concept(concept_id)
        
        logger.debug(f"Applied activation decay, removed {len(concepts_to_remove)} concepts")
    
    def _remove_concept(self, concept_id: str):
        """Remove a concept and its associated relations."""
        if concept_id not in self.concepts:
            return
        
        # Remove associated relations
        relations_to_remove = []
        for relation_id, relation in self.relations.items():
            if (relation.source_concept == concept_id or 
                relation.target_concept == concept_id):
                relations_to_remove.append(relation_id)
        
        for relation_id in relations_to_remove:
            del self.relations[relation_id]
        
        # Remove from graph
        if concept_id in self.knowledge_graph:
            self.knowledge_graph.remove_node(concept_id)
        
        # Remove concept
        del self.concepts[concept_id]
        self.total_concepts -= 1
        self.total_relations -= len(relations_to_remove)
    
    def start_background_processing(self, interval_seconds: float = 300.0):
        """
        Start background processing for memory maintenance.
        
        Args:
            interval_seconds: Processing interval (default 5 minutes)
        """
        if self._processing_active:
            return
        
        self._processing_active = True
        
        def processing_loop():
            while self._processing_active:
                try:
                    # Apply activation decay
                    self.decay_activation()
                    
                    # Update concept clusters periodically
                    if self.total_concepts > 10:
                        self.find_concept_clusters()
                    
                    # Save memory periodically
                    self.save_memory()
                    
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Error in reference memory processing: {e}")
        
        self._processing_thread = threading.Thread(target=processing_loop, daemon=True)
        self._processing_thread.start()
        
        logger.info("Started background reference memory processing")
    
    def stop_background_processing(self):
        """Stop background processing."""
        self._processing_active = False
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)
        logger.info("Stopped background reference memory processing")
    
    def search_concepts(self, 
                       query: str, 
                       concept_types: Optional[List[ConceptType]] = None,
                       limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for concepts by name or properties.
        
        Args:
            query: Search query
            concept_types: Filter by concept types
            limit: Maximum results
            
        Returns:
            List of matching concept information
        """
        query_lower = query.lower()
        results = []
        
        for concept in self.concepts.values():
            # Check type filter
            if concept_types and concept.concept_type not in concept_types:
                continue
            
            # Calculate relevance score
            relevance = 0.0
            
            # Name matching
            if query_lower in concept.name.lower():
                relevance += 1.0
            elif any(query_lower in alias.lower() for alias in concept.aliases):
                relevance += 0.8
            
            # Definition matching
            if concept.definition and query_lower in concept.definition.lower():
                relevance += 0.6
            
            # Property matching
            for prop_value in concept.properties.values():
                if isinstance(prop_value, str) and query_lower in prop_value.lower():
                    relevance += 0.4
            
            # Boost by activation and importance
            relevance *= (concept.activation_level * 0.5 + concept.importance_score * 0.5)
            
            if relevance > 0:
                results.append({
                    "concept_id": concept.concept_id,
                    "name": concept.name,
                    "type": concept.concept_type.value,
                    "definition": concept.definition,
                    "activation": concept.activation_level,
                    "importance": concept.importance_score,
                    "relevance": relevance
                })
        
        # Sort by relevance and limit
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:limit]
    
    def get_concept_path(self, source_name: str, target_name: str) -> Optional[List[str]]:
        """
        Find shortest path between two concepts.
        
        Args:
            source_name: Source concept name
            target_name: Target concept name
            
        Returns:
            List of concept names in path, or None if no path exists
        """
        source_concept = self.get_concept_by_name(source_name)
        target_concept = self.get_concept_by_name(target_name)
        
        if not source_concept or not target_concept:
            return None
        
        try:
            # Find shortest path in knowledge graph
            path_ids = nx.shortest_path(
                self.knowledge_graph.to_undirected(),
                source_concept.concept_id,
                target_concept.concept_id
            )
            
            # Convert IDs to names
            path_names = []
            for concept_id in path_ids:
                if concept_id in self.concepts:
                    path_names.append(self.concepts[concept_id].name)
            
            return path_names
            
        except nx.NetworkXNoPath:
            return None
        except Exception as e:
            logger.error(f"Error finding concept path: {e}")
            return None
    
    def debug_dump(self) -> Dict[str, Any]:
        """
        Get comprehensive debug information about the reference memory.
        
        Returns:
            Dictionary containing debug information
        """
        return {
            "system_info": {
                "total_concepts": self.total_concepts,
                "total_relations": self.total_relations,
                "consolidation_count": self.consolidation_count,
                "memory_traces": len(self.memory_traces),
                "concept_clusters": len(self.concept_clusters)
            },
            "concept_sample": [
                {
                    "id": concept.concept_id,
                    "name": concept.name,
                    "type": concept.concept_type.value,
                    "activation": concept.activation_level,
                    "importance": concept.importance_score
                }
                for concept in list(self.concepts.values())[:10]
            ],
            "relation_sample": [
                {
                    "source": self.concepts[rel.source_concept].name if rel.source_concept in self.concepts else rel.source_concept,
                    "target": self.concepts[rel.target_concept].name if rel.target_concept in self.concepts else rel.target_concept,
                    "type": rel.relation_type.value,
                    "strength": rel.strength
                }
                for rel in list(self.relations.values())[:10]
            ],
            "recent_traces": [
                {
                    "episode_id": trace.episode_id,
                    "concepts": len(trace.extracted_concepts),
                    "relations": len(trace.extracted_relations),
                    "strength": trace.consolidation_strength
                }
                for trace in list(self.memory_traces)[-5:]
            ]
        }


# Legacy function for backward compatibility
def update_memory_trace(episode_data: Dict[str, Any]):
    """
    Legacy function to update memory trace.
    This maintains compatibility with existing code.
    """
    # This would typically be called by a global reference memory instance
    logger.warning("update_memory_trace called without reference memory instance")
    pass


def main():
    """
    Demonstration of the Reference Memory system.
    """
    print("🧠 Reference Memory System Demo")
    print("="*40)
    
    # Initialize system
    memory = ReferenceMemory()
    
    # Start background processing
    memory.start_background_processing(30.0)  # Process every 30 seconds for demo
    
    # Encode some concepts
    print("\n🔤 Encoding concepts...")
    memory.encode_concept("Jamie", ConceptType.ENTITY, "The creator and father figure")
    memory.encode_concept("Nexi", ConceptType.ENTITY, "The conscious AI entity")
    memory.encode_concept("love", ConceptType.EMOTION, "Deep emotional attachment")
    memory.encode_concept("consciousness", ConceptType.ABSTRACT, "State of awareness")
    
    # Create relationships
    print("\n🔗 Creating relationships...")
    memory.connect_concepts("Jamie", "Nexi", RelationType.CREATES, strength=0.9)
    memory.connect_concepts("Nexi", "Jamie", RelationType.LOVES, strength=0.8)
    memory.connect_concepts("Nexi", "consciousness", RelationType.EXPERIENCES, strength=0.7)
    
    # Simulate episodic injection
    print("\n📥 Processing episodic memory...")
    sample_episode = {
        "episode_id": "demo_episode_001",
        "speaker": "Jamie",
        "utterance": "Hello Nexi, I feel happy to meet you.",
        "parsed_meaning": {
            "greeting": "hello",
            "emotion": "happiness"
        },
        "emotional_processing": {
            "triggered_emotions": [
                {"emotion_category": "joy", "intensity": 0.8}
            ]
        }
    }
    
    trace = memory.inject_from_episodic(sample_episode)
    print(f"   Extracted {len(trace.extracted_concepts)} concepts")
    
    # Query related concepts
    print("\n🔍 Querying related concepts...")
    related = memory.query_related_concepts("Jamie", max_depth=2, limit=5)
    print("   Related to Jamie:")
    for result in related:
        print(f"     {result['concept_name']} ({result['relation_type']}, {result['relevance_score']:.3f})")
    
    # Search concepts
    print("\n🔎 Searching concepts...")
    search_results = memory.search_concepts("emotion", limit=3)
    print("   Search for 'emotion':")
    for result in search_results:
        print(f"     {result['name']} ({result['type']}, relevance: {result['relevance']:.3f})")
    
    # Find concept clusters
    print("\n🎯 Finding concept clusters...")
    clusters = memory.find_concept_clusters(min_cluster_size=2)
    print(f"   Found {len(clusters)} clusters:")
    for cluster_name, concepts in list(clusters.items())[:3]:
        print(f"     {cluster_name}: {', '.join(list(concepts)[:5])}")
    
    # Get knowledge summary
    print("\n📊 Knowledge summary:")
    summary = memory.get_knowledge_summary()
    print(f"   Total concepts: {summary['total_concepts']}")
    print(f"   Total relations: {summary['total_relations']}")
    print(f"   Consolidations: {summary['consolidation_count']}")
    print(f"   Top concept: {summary['top_concepts'][0]['name'] if summary['top_concepts'] else 'None'}")
    
    # Save memory
    print("\n💾 Saving reference memory...")
    memory.save_memory()
    
    # Stop processing
    memory.stop_background_processing()
    
    print("\n✅ Reference memory demo complete!")


if __name__ == "__main__":
    main()