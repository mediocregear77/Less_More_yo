"""
Reality Bridge module for synchronizing and processing multi-modal sensory inputs.
Handles audio and visual perception integration with robust error handling and monitoring.
"""

import asyncio
import time
import logging
import signal
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, Callable, List, Union
from threading import Event
import statistics

from sensor_core.audio_perception import process_audio_input
from sensor_core.visual_perception import process_visual_input
from interaction_core.feedback_loop import forward_perceptual_data
from emotion_core.affective_system import update_emotional_state
from memory_core.episodic_buffer import log_experience

# Configure logging
logger = logging.getLogger(__name__)


class SensorStatus(Enum):
    """Sensor operational status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class SensorConfig:
    """Configuration for the Reality Bridge system."""
    poll_interval: float = 0.3
    max_retries: int = 3
    retry_delay: float = 0.1
    health_check_interval: float = 5.0
    metric_window_size: int = 100
    sensor_timeout: float = 1.0
    graceful_shutdown_timeout: float = 5.0
    
    # Sensor priorities (higher = more important)
    audio_priority: float = 0.8
    visual_priority: float = 0.7


@dataclass
class SensorMetrics:
    """Metrics for monitoring sensor performance."""
    total_reads: int = 0
    failed_reads: int = 0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    status: SensorStatus = SensorStatus.UNKNOWN
    
    def add_latency(self, latency: float):
        """Record a latency measurement."""
        self.latencies.append(latency)
    
    def get_avg_latency(self) -> Optional[float]:
        """Get average latency over the window."""
        if not self.latencies:
            return None
        return statistics.mean(self.latencies)
    
    def get_success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_reads == 0:
            return 0.0
        return (self.total_reads - self.failed_reads) / self.total_reads


@dataclass
class PerceptionBundle:
    """Structured perception data from all sensors."""
    timestamp: float
    audio: Optional[Dict[str, Any]]
    visual: Optional[Dict[str, Any]]
    merged_context: Dict[str, Any]
    sensor_health: Dict[str, SensorStatus]
    processing_time: float


class SensorProcessor:
    """Handles individual sensor processing with error recovery."""
    
    def __init__(self, name: str, process_func: Callable, 
                 timeout: float = 1.0, priority: float = 1.0):
        self.name = name
        self.process_func = process_func
        self.timeout = timeout
        self.priority = priority
        self.metrics = SensorMetrics()
    
    async def process(self) -> Optional[Dict[str, Any]]:
        """Process sensor data with timeout and error handling."""
        start_time = time.time()
        self.metrics.total_reads += 1
        
        try:
            # Run sensor processing with timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(self.process_func),
                timeout=self.timeout
            )
            
            # Record success
            latency = time.time() - start_time
            self.metrics.add_latency(latency)
            self.metrics.last_success = datetime.now()
            
            # Validate result
            if result and isinstance(result, dict):
                return result
            else:
                logger.warning(f"{self.name} returned invalid data: {result}")
                return None
                
        except asyncio.TimeoutError:
            logger.error(f"{self.name} timed out after {self.timeout}s")
            self.metrics.failed_reads += 1
            self.metrics.last_failure = datetime.now()
            return None
            
        except Exception as e:
            logger.error(f"{self.name} processing error: {e}")
            self.metrics.failed_reads += 1
            self.metrics.last_failure = datetime.now()
            return None
    
    def get_health_status(self) -> SensorStatus:
        """Determine sensor health based on metrics."""
        if self.metrics.total_reads == 0:
            return SensorStatus.UNKNOWN
        
        success_rate = self.metrics.get_success_rate()
        
        # Check recent failures
        if self.metrics.last_failure:
            time_since_failure = datetime.now() - self.metrics.last_failure
            if time_since_failure < timedelta(seconds=5):
                if success_rate < 0.5:
                    return SensorStatus.FAILED
                else:
                    return SensorStatus.DEGRADED
        
        # Overall health assessment
        if success_rate >= 0.95:
            return SensorStatus.HEALTHY
        elif success_rate >= 0.7:
            return SensorStatus.DEGRADED
        else:
            return SensorStatus.FAILED


class RealityBridge:
    """
    Orchestrates multi-modal sensory input processing with robust error handling,
    health monitoring, and graceful degradation.
    """
    
    def __init__(self, config: Optional[SensorConfig] = None):
        """
        Initialize Reality Bridge with configuration.
        
        Args:
            config: Configuration object for the system
        """
        self.config = config or SensorConfig()
        self.active = False
        self.shutdown_event = Event()
        
        # Initialize sensors
        self.sensors = {
            "audio": SensorProcessor(
                "audio", 
                process_audio_input,
                self.config.sensor_timeout,
                self.config.audio_priority
            ),
            "visual": SensorProcessor(
                "visual",
                process_visual_input,
                self.config.sensor_timeout,
                self.config.visual_priority
            )
        }
        
        # Processing history
        self.perception_history = deque(maxlen=1000)
        self.last_processed = None
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def synchronize_inputs(self) -> Optional[PerceptionBundle]:
        """
        Synchronize and process inputs from all sensors.
        
        Returns:
            PerceptionBundle with processed data or None if critical failure
        """
        start_time = time.time()
        
        # Process sensors concurrently
        tasks = {
            name: asyncio.create_task(sensor.process())
            for name, sensor in self.sensors.items()
        }
        
        # Wait for all sensors with individual handling
        results = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception as e:
                logger.error(f"Unexpected error in {name} sensor: {e}")
                results[name] = None
        
        # Check sensor health
        sensor_health = {
            name: sensor.get_health_status()
            for name, sensor in self.sensors.items()
        }
        
        # Determine if we have minimum viable data
        if all(status == SensorStatus.FAILED for status in sensor_health.values()):
            logger.critical("All sensors have failed, cannot process perception")
            return None
        
        # Merge available inputs
        merged_context = self.combine_inputs(results.get("audio"), results.get("visual"))
        
        # Create perception bundle
        bundle = PerceptionBundle(
            timestamp=time.time(),
            audio=results.get("audio"),
            visual=results.get("visual"),
            merged_context=merged_context,
            sensor_health=sensor_health,
            processing_time=time.time() - start_time
        )
        
        # Update history
        self.last_processed = bundle
        self.perception_history.append(bundle)
        
        # Route perception data
        await self.route_perception(bundle)
        
        return bundle
    
    def combine_inputs(self, audio: Optional[Dict[str, Any]], 
                      visual: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Intelligently merge multi-modal inputs with validation.
        
        Args:
            audio: Audio perception data
            visual: Visual perception data
            
        Returns:
            Merged context dictionary
        """
        context = {
            "timestamp": time.time(),
            "modalities_available": []
        }
        
        # Process audio data
        if audio and isinstance(audio, dict):
            context["modalities_available"].append("audio")
            audio_context = {
                "speaker": audio.get("speaker"),
                "tone": audio.get("emotion"),
                "text": audio.get("text"),
                "confidence": audio.get("confidence", 0.0)
            }
            # Filter out None values
            context["audio"] = {k: v for k, v in audio_context.items() if v is not None}
        
        # Process visual data
        if visual and isinstance(visual, dict):
            context["modalities_available"].append("visual")
            visual_context = {
                "face_id": visual.get("face_id"),
                "expression": visual.get("expression"),
                "gaze_direction": visual.get("gaze_direction"),
                "confidence": visual.get("confidence", 0.0)
            }
            # Filter out None values
            context["visual"] = {k: v for k, v in visual_context.items() if v is not None}
        
        # Cross-modal validation and enhancement
        if "audio" in context and "visual" in context:
            # Check for consistency between modalities
            context["cross_modal_confidence"] = self._calculate_cross_modal_confidence(
                context.get("audio", {}),
                context.get("visual", {})
            )
        
        return context
    
    def _calculate_cross_modal_confidence(self, audio: Dict[str, Any], 
                                        visual: Dict[str, Any]) -> float:
        """
        Calculate confidence based on cross-modal consistency.
        
        Args:
            audio: Audio context data
            visual: Visual context data
            
        Returns:
            Cross-modal confidence score
        """
        confidence = 0.5  # Base confidence
        
        # Check emotion consistency
        if audio.get("tone") and visual.get("expression"):
            # Simple mapping - in production, use more sophisticated matching
            emotion_map = {
                "happy": ["smile", "joy"],
                "sad": ["frown", "sadness"],
                "angry": ["anger", "frustration"],
                "neutral": ["neutral", "calm"]
            }
            
            for emotion, expressions in emotion_map.items():
                if audio["tone"] == emotion and visual["expression"] in expressions:
                    confidence += 0.3
                    break
        
        # Weight by individual modality confidences
        audio_conf = audio.get("confidence", 0.5)
        visual_conf = visual.get("confidence", 0.5)
        confidence *= (audio_conf + visual_conf) / 2
        
        return min(1.0, max(0.0, confidence))
    
    async def route_perception(self, bundle: PerceptionBundle):
        """
        Route perception data to appropriate subsystems.
        
        Args:
            bundle: Processed perception bundle
        """
        try:
            # Update emotional system first (highest priority)
            await asyncio.to_thread(
                update_emotional_state,
                bundle.merged_context
            )
            
            # Log to episodic buffer
            experience_data = {
                "timestamp": bundle.timestamp,
                "event": "perception_bundle",
                "details": bundle.merged_context,
                "sensor_health": {k: v.value for k, v in bundle.sensor_health.items()},
                "processing_time_ms": bundle.processing_time * 1000
            }
            await asyncio.to_thread(log_experience, experience_data)
            
            # Forward to feedback loop for interaction handling
            if bundle.merged_context.get("modalities_available"):
                await asyncio.to_thread(
                    forward_perceptual_data,
                    bundle.merged_context
                )
                
        except Exception as e:
            logger.error(f"Error routing perception data: {e}")
    
    async def health_check_loop(self):
        """Periodic health check and metrics reporting."""
        while self.active:
            await asyncio.sleep(self.config.health_check_interval)
            
            # Report sensor health
            for name, sensor in self.sensors.items():
                status = sensor.get_health_status()
                metrics = sensor.metrics
                
                logger.info(
                    f"{name} sensor health: {status.value}, "
                    f"success_rate: {metrics.get_success_rate():.2%}, "
                    f"avg_latency: {metrics.get_avg_latency():.3f}s"
                )
            
            # Check overall system health
            if all(s.get_health_status() == SensorStatus.FAILED 
                   for s in self.sensors.values()):
                logger.critical("All sensors failed, considering emergency shutdown")
    
    async def processing_loop(self):
        """Main processing loop with adaptive polling."""
        consecutive_failures = 0
        
        while self.active:
            try:
                bundle = await self.synchronize_inputs()
                
                if bundle:
                    consecutive_failures = 0
                    # Adaptive polling - speed up if things are changing rapidly
                    if self._is_high_activity(bundle):
                        poll_interval = self.config.poll_interval * 0.5
                    else:
                        poll_interval = self.config.poll_interval
                else:
                    consecutive_failures += 1
                    # Back off on failures
                    poll_interval = min(
                        self.config.poll_interval * (2 ** consecutive_failures),
                        5.0  # Max 5 second interval
                    )
                
                await asyncio.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Unexpected error in processing loop: {e}")
                await asyncio.sleep(self.config.poll_interval)
    
    def _is_high_activity(self, bundle: PerceptionBundle) -> bool:
        """
        Detect if there's high sensory activity requiring faster polling.
        
        Args:
            bundle: Current perception bundle
            
        Returns:
            True if high activity detected
        """
        # Simple heuristic - check if significant changes from last bundle
        if not self.perception_history or len(self.perception_history) < 2:
            return False
        
        prev_bundle = self.perception_history[-2]
        
        # Check for new faces, speakers, or emotional changes
        changes = []
        
        if bundle.audio and prev_bundle.audio:
            changes.append(
                bundle.audio.get("speaker") != prev_bundle.audio.get("speaker") or
                bundle.audio.get("tone") != prev_bundle.audio.get("tone")
            )
        
        if bundle.visual and prev_bundle.visual:
            changes.append(
                bundle.visual.get("face_id") != prev_bundle.visual.get("face_id") or
                bundle.visual.get("expression") != prev_bundle.visual.get("expression")
            )
        
        return any(changes)
    
    async def start(self):
        """Start the reality bridge processing."""
        if self.active:
            logger.warning("Reality bridge already running")
            return
        
        logger.info("Starting Reality Bridge")
        self.active = True
        self.shutdown_event.clear()
        
        # Create tasks for main loop and health monitoring
        tasks = [
            asyncio.create_task(self.processing_loop()),
            asyncio.create_task(self.health_check_loop())
        ]
        
        try:
            # Run until shutdown is requested
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Reality Bridge tasks cancelled")
        finally:
            # Cleanup
            for task in tasks:
                if not task.done():
                    task.cancel()
    
    def stop(self):
        """Initiate graceful shutdown."""
        logger.info("Stopping Reality Bridge")
        self.active = False
        self.shutdown_event.set()
    
    async def shutdown(self):
        """Perform graceful shutdown with timeout."""
        self.stop()
        
        # Wait for processing to complete or timeout
        start_time = time.time()
        while (time.time() - start_time < self.config.graceful_shutdown_timeout and
               len([t for t in asyncio.all_tasks() if not t.done()]) > 1):
            await asyncio.sleep(0.1)
        
        logger.info("Reality Bridge shutdown complete")
    
    def get_sensor_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get current sensor statistics."""
        return {
            name: {
                "status": sensor.get_health_status().value,
                "total_reads": sensor.metrics.total_reads,
                "failed_reads": sensor.metrics.failed_reads,
                "success_rate": sensor.metrics.get_success_rate(),
                "avg_latency": sensor.metrics.get_avg_latency(),
                "last_success": sensor.metrics.last_success.isoformat() 
                               if sensor.metrics.last_success else None,
                "last_failure": sensor.metrics.last_failure.isoformat()
                               if sensor.metrics.last_failure else None
            }
            for name, sensor in self.sensors.items()
        }


# Factory functions for different use cases
async def create_reality_bridge(config: Optional[SensorConfig] = None) -> RealityBridge:
    """
    Create and start a reality bridge instance.
    
    Args:
        config: Optional configuration
        
    Returns:
        Running RealityBridge instance
    """
    bridge = RealityBridge(config)
    asyncio.create_task(bridge.start())
    return bridge


def create_sync_reality_bridge(config: Optional[SensorConfig] = None) -> RealityBridge:
    """
    Create a reality bridge for synchronous usage.
    
    Args:
        config: Optional configuration
        
    Returns:
        RealityBridge instance
    """
    return RealityBridge(config)