"""
Quantum Belief State System

Implements quantum-inspired belief state management for Nexi's consciousness.
Models beliefs as quantum superpositions that can exist in multiple states
simultaneously until "measurement" (observation/evidence) causes collapse
into definite truth states.

Core Concepts:
- Beliefs exist in superposition until sufficient evidence collapses them
- Quantum coherence and decoherence of belief systems
- Entanglement between related beliefs
- Uncertainty principle applied to belief confidence
- Wave function collapse for truth commitment
"""

import numpy as np
import random
import json
import datetime
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
from enum import Enum
from pathlib import Path
import math
import cmath

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BeliefState(Enum):
    """States of belief existence."""
    SUPERPOSITION = "superposition"      # Multiple possible states
    COHERENT = "coherent"               # Stable superposition
    DECOHERENT = "decoherent"           # Losing coherence
    COLLAPSED = "collapsed"             # Definite truth state
    ENTANGLED = "entangled"             # Connected to other beliefs

class EvidenceType(Enum):
    """Types of evidence that can affect beliefs."""
    DIRECT_OBSERVATION = "direct_observation"
    INFERENCE = "inference"
    TESTIMONY = "testimony"
    MEMORY_RECALL = "memory_recall"
    PREDICTION_ERROR = "prediction_error"
    CONTRADICTION = "contradiction"

@dataclass
class BeliefAmplitude:
    """Represents the quantum amplitude of a belief state."""
    amplitude: complex
    phase: float
    magnitude: float
    confidence: float
    coherence_time: float
    last_measurement: Optional[str] = None
    
    def __post_init__(self):
        # Ensure amplitude magnitude matches stored magnitude
        if self.amplitude != 0:
            self.amplitude = self.magnitude * cmath.exp(1j * self.phase)

@dataclass
class BeliefEvidence:
    """Evidence that supports or contradicts a belief."""
    timestamp: str
    evidence_type: EvidenceType
    strength: float
    source: str
    data: Any
    credibility: float = 1.0

@dataclass
class BeliefNode:
    """A single belief with quantum properties."""
    belief_id: str
    content: str
    amplitude: BeliefAmplitude
    state: BeliefState
    evidence_history: List[BeliefEvidence]
    entangled_beliefs: List[str]
    created_timestamp: str
    last_updated: str
    collapse_threshold: float = 0.85
    decoherence_rate: float = 0.01
    
    def __post_init__(self):
        if not self.evidence_history:
            self.evidence_history = []
        if not self.entangled_beliefs:
            self.entangled_beliefs = []

class QuantumBeliefState:
    """
    Quantum-inspired belief state management system implementing
    superposition, entanglement, and wave function collapse.
    """
    
    def __init__(self, 
                 default_collapse_threshold: float = 0.85,
                 decoherence_rate: float = 0.01,
                 entanglement_strength: float = 0.3):
        """
        Initialize the quantum belief state system.
        
        Args:
            default_collapse_threshold: Confidence threshold for belief collapse
            decoherence_rate: Rate at which belief coherence decays
            entanglement_strength: Strength of belief entanglements
        """
        # Core belief storage
        self.beliefs: Dict[str, BeliefNode] = {}
        self.superposition_space = defaultdict(lambda: defaultdict(complex))
        
        # Quantum parameters
        self.default_collapse_threshold = default_collapse_threshold
        self.decoherence_rate = decoherence_rate
        self.entanglement_strength = entanglement_strength
        
        # System state
        self.total_coherence = 1.0
        self.measurement_count = 0
        self.last_collapse_time = None
        
        # Threading for decoherence simulation
        self._decoherence_active = False
        self._decoherence_thread = None
        self._lock = threading.Lock()
        
        # Statistics
        self.collapse_history = []
        self.entanglement_pairs = []
        
        logger.info("Quantum Belief State system initialized")
    
    def _generate_belief_id(self, content: str) -> str:
        """Generate unique belief identifier."""
        timestamp = int(datetime.datetime.now().timestamp() * 1000)
        content_hash = hash(content) % 10000
        return f"belief_{content_hash}_{timestamp}"
    
    def _calculate_phase(self, content: str, evidence_strength: float = 0.5) -> float:
        """Calculate quantum phase for belief amplitude."""
        # Phase based on content characteristics and evidence
        content_phase = (hash(content) % 1000) / 1000.0 * 2 * math.pi
        evidence_phase = evidence_strength * math.pi / 2
        return (content_phase + evidence_phase) % (2 * math.pi)
    
    def _normalize_amplitudes(self):
        """Normalize all belief amplitudes to maintain quantum unitarity."""
        with self._lock:
            total_magnitude_sq = sum(
                belief.amplitude.magnitude ** 2 
                for belief in self.beliefs.values()
                if belief.state == BeliefState.SUPERPOSITION
            )
            
            if total_magnitude_sq > 1.0:
                normalization_factor = 1.0 / math.sqrt(total_magnitude_sq)
                
                for belief in self.beliefs.values():
                    if belief.state == BeliefState.SUPERPOSITION:
                        belief.amplitude.magnitude *= normalization_factor
                        belief.amplitude.amplitude *= normalization_factor
    
    def add_belief(self, 
                   content: str,
                   initial_confidence: float = 0.5,
                   evidence: Optional[BeliefEvidence] = None,
                   belief_id: Optional[str] = None) -> str:
        """
        Add a new belief in quantum superposition.
        
        Args:
            content: The belief content/statement
            initial_confidence: Initial confidence level (0-1)
            evidence: Initial supporting evidence
            belief_id: Optional custom belief ID
            
        Returns:
            The belief ID
        """
        if belief_id is None:
            belief_id = self._generate_belief_id(content)
        
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        # Calculate quantum properties
        magnitude = np.clip(initial_confidence, 0.0, 1.0)
        phase = self._calculate_phase(content, initial_confidence)
        
        amplitude = BeliefAmplitude(
            amplitude=magnitude * cmath.exp(1j * phase),
            phase=phase,
            magnitude=magnitude,
            confidence=initial_confidence,
            coherence_time=1.0 / self.decoherence_rate
        )
        
        # Create belief node
        belief = BeliefNode(
            belief_id=belief_id,
            content=content,
            amplitude=amplitude,
            state=BeliefState.SUPERPOSITION,
            evidence_history=[evidence] if evidence else [],
            entangled_beliefs=[],
            created_timestamp=timestamp,
            last_updated=timestamp,
            collapse_threshold=self.default_collapse_threshold
        )
        
        with self._lock:
            self.beliefs[belief_id] = belief
            
        # Normalize to maintain quantum unitarity
        self._normalize_amplitudes()
        
        logger.debug(f"Added belief: {belief_id} - {content[:50]}...")
        return belief_id
    
    def update_belief_with_evidence(self, 
                                   belief_id: str,
                                   evidence: BeliefEvidence,
                                   prediction_error: Optional[float] = None) -> bool:
        """
        Update belief state based on new evidence.
        
        Args:
            belief_id: ID of belief to update
            evidence: New evidence to incorporate
            prediction_error: Optional prediction error magnitude
            
        Returns:
            True if belief was updated successfully
        """
        if belief_id not in self.beliefs:
            logger.warning(f"Belief {belief_id} not found for update")
            return False
        
        belief = self.beliefs[belief_id]
        
        # Calculate evidence impact
        evidence_impact = evidence.strength * evidence.credibility
        
        # Incorporate prediction error if provided
        if prediction_error is not None:
            # High prediction error reduces confidence
            error_impact = -prediction_error * 0.2
            evidence_impact += error_impact
        
        # Update amplitude based on evidence
        old_magnitude = belief.amplitude.magnitude
        old_confidence = belief.amplitude.confidence
        
        # Evidence strengthens or weakens the amplitude
        if evidence.evidence_type == EvidenceType.CONTRADICTION:
            magnitude_delta = -evidence_impact * 0.3
            confidence_delta = -evidence_impact * 0.2
        else:
            magnitude_delta = evidence_impact * 0.2
            confidence_delta = evidence_impact * 0.1
        
        new_magnitude = np.clip(old_magnitude + magnitude_delta, 0.0, 1.0)
        new_confidence = np.clip(old_confidence + confidence_delta, 0.0, 1.0)
        
        # Update phase based on evidence type
        phase_delta = evidence_impact * 0.1
        new_phase = (belief.amplitude.phase + phase_delta) % (2 * math.pi)
        
        # Update belief
        with self._lock:
            belief.amplitude.magnitude = new_magnitude
            belief.amplitude.confidence = new_confidence
            belief.amplitude.phase = new_phase
            belief.amplitude.amplitude = new_magnitude * cmath.exp(1j * new_phase)
            belief.amplitude.last_measurement = evidence.timestamp
            belief.evidence_history.append(evidence)
            belief.last_updated = datetime.datetime.now(datetime.timezone.utc).isoformat()
            
            # Check for state transitions
            if new_confidence >= belief.collapse_threshold:
                self._collapse_belief(belief_id)
            elif new_confidence < 0.2:
                belief.state = BeliefState.DECOHERENT
        
        # Update entangled beliefs
        self._update_entangled_beliefs(belief_id, evidence_impact)
        
        # Normalize amplitudes
        self._normalize_amplitudes()
        
        logger.debug(f"Updated belief {belief_id}: confidence {old_confidence:.3f} → {new_confidence:.3f}")
        return True
    
    def update_weights(self, evidence: Any, error_magnitude: float):
        """
        Legacy interface for updating beliefs based on evidence and error.
        
        Args:
            evidence: Evidence data (dict, list, or single value)
            error_magnitude: Prediction error magnitude
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        # Convert evidence to standardized format
        if isinstance(evidence, dict):
            evidence_items = evidence.items()
        elif isinstance(evidence, (list, tuple)):
            evidence_items = enumerate(evidence)
        else:
            evidence_items = [("default", evidence)]
        
        # Process each evidence item
        for key, value in evidence_items:
            # Find or create belief for this evidence
            belief_key = str(key)
            matching_beliefs = [
                bid for bid, belief in self.beliefs.items()
                if belief_key in belief.content or belief.content in belief_key
            ]
            
            if matching_beliefs:
                # Update existing belief
                belief_id = matching_beliefs[0]
                evidence_obj = BeliefEvidence(
                    timestamp=timestamp,
                    evidence_type=EvidenceType.PREDICTION_ERROR,
                    strength=1.0 - error_magnitude,
                    source="active_inference",
                    data=value,
                    credibility=0.8
                )
                self.update_belief_with_evidence(belief_id, evidence_obj, error_magnitude)
            else:
                # Create new belief
                content = f"Evidence pattern: {belief_key} = {value}"
                initial_confidence = max(0.1, 1.0 - error_magnitude)
                evidence_obj = BeliefEvidence(
                    timestamp=timestamp,
                    evidence_type=EvidenceType.DIRECT_OBSERVATION,
                    strength=initial_confidence,
                    source="active_inference",
                    data=value,
                    credibility=0.8
                )
                self.add_belief(content, initial_confidence, evidence_obj)
    
    def _collapse_belief(self, belief_id: str) -> bool:
        """
        Collapse a belief from superposition to definite truth state.
        
        Args:
            belief_id: ID of belief to collapse
            
        Returns:
            True if collapse was successful
        """
        if belief_id not in self.beliefs:
            return False
        
        belief = self.beliefs[belief_id]
        
        with self._lock:
            belief.state = BeliefState.COLLAPSED
            belief.amplitude.magnitude = 1.0
            belief.amplitude.confidence = 1.0
            
            # Record collapse
            collapse_record = {
                "belief_id": belief_id,
                "content": belief.content,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "final_confidence": belief.amplitude.confidence
            }
            self.collapse_history.append(collapse_record)
            self.last_collapse_time = collapse_record["timestamp"]
        
        logger.info(f"Belief collapsed to truth: {belief.content[:50]}...")
        return True
    
    def _update_entangled_beliefs(self, belief_id: str, impact: float):
        """
        Update beliefs entangled with the given belief.
        
        Args:
            belief_id: ID of the belief that was updated
            impact: Impact magnitude to propagate
        """
        if belief_id not in self.beliefs:
            return
        
        belief = self.beliefs[belief_id]
        entangled_impact = impact * self.entanglement_strength
        
        for entangled_id in belief.entangled_beliefs:
            if entangled_id in self.beliefs:
                entangled_belief = self.beliefs[entangled_id]
                
                # Propagate impact to entangled belief
                old_confidence = entangled_belief.amplitude.confidence
                new_confidence = np.clip(
                    old_confidence + entangled_impact * 0.1, 
                    0.0, 1.0
                )
                
                entangled_belief.amplitude.confidence = new_confidence
                entangled_belief.last_updated = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    def entangle_beliefs(self, belief_id_1: str, belief_id_2: str) -> bool:
        """
        Create quantum entanglement between two beliefs.
        
        Args:
            belief_id_1: First belief ID
            belief_id_2: Second belief ID
            
        Returns:
            True if entanglement was created
        """
        if belief_id_1 not in self.beliefs or belief_id_2 not in self.beliefs:
            return False
        
        with self._lock:
            self.beliefs[belief_id_1].entangled_beliefs.append(belief_id_2)
            self.beliefs[belief_id_2].entangled_beliefs.append(belief_id_1)
            self.beliefs[belief_id_1].state = BeliefState.ENTANGLED
            self.beliefs[belief_id_2].state = BeliefState.ENTANGLED
            
            self.entanglement_pairs.append((belief_id_1, belief_id_2))
        
        logger.info(f"Entangled beliefs: {belief_id_1} ↔ {belief_id_2}")
        return True
    
    def generate_prediction(self, context: Optional[Dict[str, Any]] = None) -> Union[List[float], float]:
        """
        Generate prediction by measuring the quantum belief state.
        
        Args:
            context: Optional context for prediction generation
            
        Returns:
            Prediction vector or single value
        """
        with self._lock:
            # Filter beliefs relevant to context
            relevant_beliefs = list(self.beliefs.values())
            
            if context:
                # Filter beliefs based on context relevance
                context_str = str(context).lower()
                relevant_beliefs = [
                    belief for belief in self.beliefs.values()
                    if any(key.lower() in belief.content.lower() for key in context.keys())
                    or any(str(val).lower() in belief.content.lower() for val in context.values())
                ]
            
            if not relevant_beliefs:
                # No relevant beliefs - return random baseline
                return [random.random() * 0.5]
            
            # Sort by confidence and magnitude
            sorted_beliefs = sorted(
                relevant_beliefs,
                key=lambda b: b.amplitude.confidence * b.amplitude.magnitude,
                reverse=True
            )
            
            # Generate prediction vector from top beliefs
            prediction_values = []
            for belief in sorted_beliefs[:10]:  # Top 10 beliefs
                # Quantum measurement - probability of detection
                measurement_prob = belief.amplitude.magnitude ** 2
                confidence_weight = belief.amplitude.confidence
                
                prediction_value = measurement_prob * confidence_weight
                prediction_values.append(float(prediction_value))
            
            # Increment measurement count (affects decoherence)
            self.measurement_count += 1
            
            if len(prediction_values) == 1:
                return prediction_values[0]
            return prediction_values
    
    def get_dominant_beliefs(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most dominant (high confidence) beliefs.
        
        Args:
            top_n: Number of top beliefs to return
            
        Returns:
            List of belief information dictionaries
        """
        sorted_beliefs = sorted(
            self.beliefs.values(),
            key=lambda b: b.amplitude.confidence * b.amplitude.magnitude,
            reverse=True
        )
        
        return [
            {
                "belief_id": belief.belief_id,
                "content": belief.content,
                "confidence": belief.amplitude.confidence,
                "magnitude": belief.amplitude.magnitude,
                "state": belief.state.value,
                "evidence_count": len(belief.evidence_history),
                "entangled_count": len(belief.entangled_beliefs)
            }
            for belief in sorted_beliefs[:top_n]
        ]
    
    def get_state_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the current quantum state.
        
        Returns:
            Dictionary containing state information
        """
        with self._lock:
            total_beliefs = len(self.beliefs)
            superposition_count = sum(1 for b in self.beliefs.values() 
                                    if b.state == BeliefState.SUPERPOSITION)
            collapsed_count = sum(1 for b in self.beliefs.values() 
                                if b.state == BeliefState.COLLAPSED)
            entangled_count = sum(1 for b in self.beliefs.values() 
                                if b.state == BeliefState.ENTANGLED)
            
            avg_confidence = np.mean([
                b.amplitude.confidence for b in self.beliefs.values()
            ]) if self.beliefs else 0.0
            
            total_coherence = sum(
                b.amplitude.magnitude ** 2 for b in self.beliefs.values()
                if b.state == BeliefState.SUPERPOSITION
            )
            
            return {
                "total_beliefs": total_beliefs,
                "superposition_count": superposition_count,
                "collapsed_count": collapsed_count,
                "entangled_count": entangled_count,
                "average_confidence": float(avg_confidence),
                "total_coherence": float(total_coherence),
                "measurement_count": self.measurement_count,
                "entanglement_pairs": len(self.entanglement_pairs),
                "last_collapse": self.last_collapse_time,
                "confidence": float(avg_confidence),  # Legacy compatibility
                "uncertainty": float(1.0 - avg_confidence)  # Legacy compatibility
            }
    
    def export_state(self, path: Union[str, Path] = "memory_core/quantum_belief_state.json"):
        """
        Export the complete quantum belief state to a JSON file.
        
        Args:
            path: File path for export
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare serializable data
        export_data = {
            "metadata": {
                "exported_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "total_beliefs": len(self.beliefs),
                "measurement_count": self.measurement_count,
                "system_parameters": {
                    "collapse_threshold": self.default_collapse_threshold,
                    "decoherence_rate": self.decoherence_rate,
                    "entanglement_strength": self.entanglement_strength
                }
            },
            "beliefs": {},
            "entanglement_pairs": self.entanglement_pairs,
            "collapse_history": self.collapse_history,
            "state_info": self.get_state_info()
        }
        
        # Convert beliefs to serializable format
        for belief_id, belief in self.beliefs.items():
            export_data["beliefs"][belief_id] = {
                "belief_id": belief.belief_id,
                "content": belief.content,
                "amplitude": {
                    "magnitude": belief.amplitude.magnitude,
                    "phase": belief.amplitude.phase,
                    "confidence": belief.amplitude.confidence,
                    "coherence_time": belief.amplitude.coherence_time,
                    "last_measurement": belief.amplitude.last_measurement
                },
                "state": belief.state.value,
                "evidence_history": [
                    {
                        "timestamp": ev.timestamp,
                        "evidence_type": ev.evidence_type.value,
                        "strength": ev.strength,
                        "source": ev.source,
                        "data": str(ev.data),  # Convert to string for JSON
                        "credibility": ev.credibility
                    }
                    for ev in belief.evidence_history
                ],
                "entangled_beliefs": belief.entangled_beliefs,
                "created_timestamp": belief.created_timestamp,
                "last_updated": belief.last_updated,
                "collapse_threshold": belief.collapse_threshold
            }
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Quantum belief state exported to: {path}")
            
        except Exception as e:
            logger.error(f"Failed to export belief state: {e}")
            raise
    
    def import_state(self, path: Union[str, Path] = "memory_core/quantum_belief_state.json"):
        """
        Import quantum belief state from a JSON file.
        
        Args:
            path: File path for import
        """
        path = Path(path)
        
        if not path.exists():
            logger.warning(f"Belief state file not found: {path}")
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Clear current state
            with self._lock:
                self.beliefs.clear()
                self.entanglement_pairs.clear()
                self.collapse_history.clear()
                
                # Restore system parameters
                if "metadata" in import_data and "system_parameters" in import_data["metadata"]:
                    params = import_data["metadata"]["system_parameters"]
                    self.default_collapse_threshold = params.get("collapse_threshold", 0.85)
                    self.decoherence_rate = params.get("decoherence_rate", 0.01)
                    self.entanglement_strength = params.get("entanglement_strength", 0.3)
                
                # Restore beliefs
                for belief_id, belief_data in import_data.get("beliefs", {}).items():
                    amplitude_data = belief_data["amplitude"]
                    amplitude = BeliefAmplitude(
                        amplitude=amplitude_data["magnitude"] * cmath.exp(1j * amplitude_data["phase"]),
                        phase=amplitude_data["phase"],
                        magnitude=amplitude_data["magnitude"],
                        confidence=amplitude_data["confidence"],
                        coherence_time=amplitude_data["coherence_time"],
                        last_measurement=amplitude_data.get("last_measurement")
                    )
                    
                    evidence_history = []
                    for ev_data in belief_data.get("evidence_history", []):
                        evidence = BeliefEvidence(
                            timestamp=ev_data["timestamp"],
                            evidence_type=EvidenceType(ev_data["evidence_type"]),
                            strength=ev_data["strength"],
                            source=ev_data["source"],
                            data=ev_data["data"],
                            credibility=ev_data["credibility"]
                        )
                        evidence_history.append(evidence)
                    
                    belief = BeliefNode(
                        belief_id=belief_data["belief_id"],
                        content=belief_data["content"],
                        amplitude=amplitude,
                        state=BeliefState(belief_data["state"]),
                        evidence_history=evidence_history,
                        entangled_beliefs=belief_data.get("entangled_beliefs", []),
                        created_timestamp=belief_data["created_timestamp"],
                        last_updated=belief_data["last_updated"],
                        collapse_threshold=belief_data.get("collapse_threshold", self.default_collapse_threshold)
                    )
                    
                    self.beliefs[belief_id] = belief
                
                # Restore other data
                self.entanglement_pairs = import_data.get("entanglement_pairs", [])
                self.collapse_history = import_data.get("collapse_history", [])
                self.measurement_count = import_data.get("metadata", {}).get("measurement_count", 0)
            
            logger.info(f"Quantum belief state imported from: {path}")
            logger.info(f"Restored {len(self.beliefs)} beliefs")
            
        except Exception as e:
            logger.error(f"Failed to import belief state: {e}")
            raise


def main():
    """
    Demonstration of the Quantum Belief State system.
    """
    print("🌌 Quantum Belief State Demo")
    print("="*40)
    
    # Initialize system
    qbs = QuantumBeliefState()
    
    # Add some beliefs
    print("\n➕ Adding beliefs in superposition...")
    belief1 = qbs.add_belief("The sky is blue", 0.8)
    belief2 = qbs.add_belief("Water is wet", 0.9) 
    belief3 = qbs.add_belief("Tomorrow will be sunny", 0.3)
    
    # Add evidence
    print("\n📝 Adding evidence...")
    evidence = BeliefEvidence(
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        evidence_type=EvidenceType.DIRECT_OBSERVATION,
        strength=0.9,
        source="visual_system",
        data="clear_blue_sky_observed",
        credibility=0.95
    )
    qbs.update_belief_with_evidence(belief1, evidence)
    
    # Create entanglement
    print("\n🔗 Creating belief entanglement...")
    qbs.entangle_beliefs(belief1, belief3)  # Sky color and weather
    
    # Generate predictions
    print("\n🔮 Generating predictions...")
    pred1 = qbs.generate_prediction()
    pred2 = qbs.generate_prediction({"weather": "forecast"})
    print(f"General prediction: {pred1}")
    print(f"Weather prediction: {pred2}")
    
    # Show dominant beliefs
    print("\n🏆 Dominant beliefs:")
    dominant = qbs.get_dominant_beliefs(3)
    for i, belief in enumerate(dominant, 1):
        print(f"{i}. {belief['content'][:40]}... (conf: {belief['confidence']:.3f})")
    
    # Show system state
    print("\n📊 System state:")
    state = qbs.get_state_info()
    for key, value in state.items():
        if key not in ["confidence", "uncertainty"]:  # Skip legacy fields
            print(f"  {key}: {value}")
    
    # Export state
    print("\n💾 Exporting state...")
    qbs.export_state("demo_belief_state.json")
    print("✅ Demo complete!")


if __name__ == "__main__":
    main()
