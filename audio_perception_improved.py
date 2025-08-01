"""
Audio perception module with speech recognition, emotion analysis, and speaker identification.
Provides robust audio processing with VAD, noise reduction, and health monitoring.
"""

import asyncio
import logging
import queue
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, Tuple, List, Callable
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import speech_recognition as sr
import librosa
import sounddevice as sd
import webrtcvad
from scipy import signal

from emotion_core.affective_system import interpret_audio_emotion

# Suppress librosa warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logger = logging.getLogger(__name__)


class AudioState(Enum):
    """Audio processing state."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    ERROR = "error"


@dataclass
class AudioConfig:
    """Configuration for audio perception system."""
    # Audio parameters
    sampling_rate: int = 16000
    channels: int = 1
    chunk_duration: float = 0.5  # seconds per chunk for VAD
    buffer_duration: float = 4.0  # seconds for full utterance
    
    # Queue management
    max_queue_size: int = 100
    
    # Voice Activity Detection
    vad_aggressiveness: int = 2  # 0-3, higher = more aggressive
    vad_frame_duration: int = 30  # ms (10, 20, or 30)
    speech_threshold: float = 0.7  # proportion of frames with speech
    
    # Noise reduction
    noise_reduction_strength: float = 0.8
    
    # Recognition
    recognition_language: str = "en-US"
    recognition_timeout: float = 5.0
    use_offline_recognition: bool = False
    
    # Emotion analysis
    n_mfcc: int = 13
    n_mels: int = 128
    
    # Performance
    thread_pool_size: int = 3


@dataclass
class AudioMetrics:
    """Metrics for audio processing performance."""
    total_chunks: int = 0
    speech_chunks: int = 0
    transcriptions_success: int = 0
    transcriptions_failed: int = 0
    emotion_analyses: int = 0
    avg_processing_time: deque = field(default_factory=lambda: deque(maxlen=100))
    last_error: Optional[str] = None
    
    def add_processing_time(self, duration: float):
        """Record processing time."""
        self.avg_processing_time.append(duration)
    
    def get_avg_processing_time(self) -> float:
        """Get average processing time."""
        if not self.avg_processing_time:
            return 0.0
        return sum(self.avg_processing_time) / len(self.avg_processing_time)


class VoiceActivityDetector:
    """Voice Activity Detection using WebRTC VAD."""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.vad = webrtcvad.Vad(config.vad_aggressiveness)
        self.frame_duration_ms = config.vad_frame_duration
        self.frame_size = int(config.sampling_rate * self.frame_duration_ms / 1000)
        
    def is_speech(self, audio_data: np.ndarray) -> bool:
        """
        Detect if audio contains speech.
        
        Args:
            audio_data: Audio samples
            
        Returns:
            True if speech detected
        """
        # Convert to 16-bit PCM
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Process in frames
        num_frames = len(audio_int16) // self.frame_size
        speech_frames = 0
        
        for i in range(num_frames):
            start = i * self.frame_size
            end = start + self.frame_size
            frame = audio_int16[start:end].tobytes()
            
            try:
                if self.vad.is_speech(frame, self.config.sampling_rate):
                    speech_frames += 1
            except Exception as e:
                logger.warning(f"VAD error: {e}")
                
        # Calculate speech ratio
        speech_ratio = speech_frames / max(num_frames, 1)
        return speech_ratio >= self.config.speech_threshold


class NoiseReducer:
    """Audio noise reduction using spectral subtraction."""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.noise_profile = None
        
    def update_noise_profile(self, noise_audio: np.ndarray):
        """Update noise profile from background audio."""
        # Compute noise spectrum
        _, _, Sxx = signal.spectrogram(noise_audio, fs=self.config.sampling_rate)
        self.noise_profile = np.mean(Sxx, axis=1)
        
    def reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction to audio.
        
        Args:
            audio: Input audio
            
        Returns:
            Noise-reduced audio
        """
        if self.noise_profile is None:
            # Initialize with first chunk assumed as noise
            self.update_noise_profile(audio[:self.config.sampling_rate])
            
        # Spectral subtraction
        f, t, Sxx = signal.spectrogram(audio, fs=self.config.sampling_rate)
        
        # Subtract noise profile
        Sxx_clean = Sxx - self.noise_profile[:, np.newaxis] * self.config.noise_reduction_strength
        Sxx_clean = np.maximum(Sxx_clean, 0)  # Ensure non-negative
        
        # Reconstruct signal
        _, audio_clean = signal.istft(Sxx_clean, fs=self.config.sampling_rate)
        
        return audio_clean.astype(np.float32)


class SpeechRecognizer:
    """Enhanced speech recognition with multiple backends."""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        
        # Initialize offline recognition if requested
        if config.use_offline_recognition:
            try:
                import vosk
                model_path = os.environ.get("VOSK_MODEL_PATH", "models/vosk-model-small-en-us")
                self.offline_model = vosk.Model(model_path)
                self.offline_recognizer = vosk.KaldiRecognizer(
                    self.offline_model, 
                    config.sampling_rate
                )
                logger.info("Offline speech recognition initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize offline recognition: {e}")
                self.offline_recognizer = None
        else:
            self.offline_recognizer = None
    
    async def transcribe(self, audio_np: np.ndarray) -> Tuple[str, float]:
        """
        Transcribe speech from audio with confidence score.
        
        Args:
            audio_np: Audio data as numpy array
            
        Returns:
            Tuple of (transcription, confidence)
        """
        # Convert to AudioData format
        audio_bytes = (audio_np * 32767).astype(np.int16).tobytes()
        audio_data = sr.AudioData(audio_bytes, self.config.sampling_rate, 2)
        
        try:
            # Try online recognition first
            if not self.config.use_offline_recognition:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.recognizer.recognize_google(
                        audio_data,
                        language=self.config.recognition_language,
                        show_all=True
                    )
                )
                
                if result and "alternative" in result:
                    best = result["alternative"][0]
                    return best["transcript"], best.get("confidence", 0.8)
                else:
                    return "", 0.0
                    
            # Fallback to offline recognition
            elif self.offline_recognizer:
                self.offline_recognizer.AcceptWaveform(audio_bytes)
                result = json.loads(self.offline_recognizer.Result())
                return result.get("text", ""), result.get("confidence", 0.5)
                
        except sr.UnknownValueError:
            logger.debug("Speech not recognized")
            return "", 0.0
        except sr.RequestError as e:
            logger.error(f"Recognition service error: {e}")
            return "[service unavailable]", 0.0
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return "[error]", 0.0
            
        return "", 0.0


class EmotionAnalyzer:
    """Advanced emotion analysis from audio features."""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        
    def extract_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive audio features for emotion analysis.
        
        Args:
            audio: Audio data
            
        Returns:
            Dictionary of audio features
        """
        features = {}
        
        try:
            # MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.config.sampling_rate,
                n_mfcc=self.config.n_mfcc
            )
            features["mfcc_mean"] = np.mean(mfcc, axis=1)
            features["mfcc_std"] = np.std(mfcc, axis=1)
            
            # Prosodic features
            # Pitch (fundamental frequency)
            f0, voiced_flag, _ = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.config.sampling_rate
            )
            features["pitch_mean"] = np.nanmean(f0) if f0 is not None else 0
            features["pitch_std"] = np.nanstd(f0) if f0 is not None else 0
            
            # Energy/RMS
            rms = librosa.feature.rms(y=audio)
            features["energy_mean"] = np.mean(rms)
            features["energy_std"] = np.std(rms)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio, 
                sr=self.config.sampling_rate
            )
            features["spectral_centroid_mean"] = np.mean(spectral_centroids)
            
            # Zero crossing rate (indicates speech/music characteristics)
            zcr = librosa.feature.zero_crossing_rate(audio)
            features["zcr_mean"] = np.mean(zcr)
            
            # Tempo (speech rate indicator)
            tempo, _ = librosa.beat.beat_track(y=audio, sr=self.config.sampling_rate)
            features["tempo"] = tempo
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            # Return minimal features on error
            features = {
                "mfcc_mean": np.zeros(self.config.n_mfcc),
                "mfcc_std": np.zeros(self.config.n_mfcc)
            }
            
        return features
    
    async def analyze_emotion(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Analyze emotion from audio.
        
        Args:
            audio: Audio data
            
        Returns:
            Emotion analysis results
        """
        # Extract features
        features = await asyncio.get_event_loop().run_in_executor(
            None, self.extract_features, audio
        )
        
        # Call emotion interpretation with enhanced features
        emotion_result = await asyncio.get_event_loop().run_in_executor(
            None, interpret_audio_emotion, features
        )
        
        return emotion_result


class AudioPerceptionSystem:
    """
    Main audio perception system with integrated components.
    """
    
    def __init__(self, config: Optional[AudioConfig] = None):
        """
        Initialize audio perception system.
        
        Args:
            config: System configuration
        """
        self.config = config or AudioConfig()
        self.state = AudioState.IDLE
        self.metrics = AudioMetrics()
        
        # Audio queue with size limit
        self.audio_queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        
        # Initialize components
        self.vad = VoiceActivityDetector(self.config)
        self.noise_reducer = NoiseReducer(self.config)
        self.recognizer = SpeechRecognizer(self.config)
        self.emotion_analyzer = EmotionAnalyzer(self.config)
        
        # Thread pool for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=self.config.thread_pool_size)
        
        # Audio buffer for accumulating speech
        self.audio_buffer = deque(
            maxlen=int(self.config.buffer_duration / self.config.chunk_duration)
        )
        
        # Recording state
        self.recording = False
        self.stream = None
        
    async def start_recording(self):
        """Start audio recording."""
        if self.recording:
            logger.warning("Audio recording already active")
            return
            
        self.recording = True
        self.state = AudioState.LISTENING
        
        # Start recording in background thread
        asyncio.create_task(self._recording_loop())
        logger.info("Audio recording started")
        
    async def stop_recording(self):
        """Stop audio recording."""
        self.recording = False
        self.state = AudioState.IDLE
        
        # Close stream if exists
        if self.stream:
            self.stream.close()
            self.stream = None
            
        logger.info("Audio recording stopped")
        
    async def _recording_loop(self):
        """Background recording loop."""
        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio stream status: {status}")
                self.metrics.last_error = str(status)
                
            # Don't block the audio thread
            try:
                audio_chunk = indata.copy().flatten()
                asyncio.create_task(self._process_audio_chunk(audio_chunk))
            except Exception as e:
                logger.error(f"Audio callback error: {e}")
        
        try:
            # Open audio stream
            self.stream = sd.InputStream(
                samplerate=self.config.sampling_rate,
                channels=self.config.channels,
                callback=audio_callback,
                blocksize=int(self.config.sampling_rate * self.config.chunk_duration)
            )
            
            self.stream.start()
            
            # Keep stream alive
            while self.recording:
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Recording error: {e}")
            self.state = AudioState.ERROR
            self.metrics.last_error = str(e)
        finally:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
                
    async def _process_audio_chunk(self, audio_chunk: np.ndarray):
        """Process individual audio chunk."""
        self.metrics.total_chunks += 1
        
        # Check for speech using VAD
        is_speech = await asyncio.get_event_loop().run_in_executor(
            self.executor, self.vad.is_speech, audio_chunk
        )
        
        if is_speech:
            self.metrics.speech_chunks += 1
            # Add to buffer
            self.audio_buffer.append(audio_chunk)
            
            # Check if we have enough audio for processing
            if len(self.audio_buffer) >= self.audio_buffer.maxlen:
                # Combine buffer into single array
                full_audio = np.concatenate(list(self.audio_buffer))
                
                # Queue for processing if not full
                try:
                    await asyncio.wait_for(
                        self.audio_queue.put(full_audio),
                        timeout=0.1  # Don't block recording
                    )
                    # Clear buffer after queuing
                    self.audio_buffer.clear()
                except asyncio.TimeoutError:
                    logger.warning("Audio queue full, dropping audio")
                    
    async def process_audio(self) -> Optional[Dict[str, Any]]:
        """
        Process queued audio and return perception data.
        
        Returns:
            Perception data or None if no audio available
        """
        try:
            # Get audio from queue (non-blocking)
            audio = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None
            
        start_time = time.time()
        self.state = AudioState.PROCESSING
        
        try:
            # Apply noise reduction
            audio_clean = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.noise_reducer.reduce_noise, audio
            )
            
            # Run transcription and emotion analysis in parallel
            transcription_task = self.recognizer.transcribe(audio_clean)
            emotion_task = self.emotion_analyzer.analyze_emotion(audio_clean)
            
            # Wait for both
            (text, confidence), emotion_data = await asyncio.gather(
                transcription_task, emotion_task
            )
            
            # Update metrics
            if text and text not in ["", "[error]", "[service unavailable]"]:
                self.metrics.transcriptions_success += 1
            else:
                self.metrics.transcriptions_failed += 1
                
            self.metrics.emotion_analyses += 1
            
            # Record processing time
            processing_time = time.time() - start_time
            self.metrics.add_processing_time(processing_time)
            
            # Build result
            result = {
                "text": text,
                "confidence": confidence,
                "emotion": emotion_data.get("emotion", "neutral"),
                "emotion_confidence": emotion_data.get("confidence", 0.0),
                "emotion_features": emotion_data.get("features", {}),
                "speaker": await self._identify_speaker(audio_clean),
                "processing_time": processing_time,
                "timestamp": time.time()
            }
            
            self.state = AudioState.LISTENING
            return result
            
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            self.metrics.last_error = str(e)
            self.state = AudioState.ERROR
            return None
            
    async def _identify_speaker(self, audio: np.ndarray) -> str:
        """
        Identify speaker from audio (placeholder for speaker diarization).
        
        Args:
            audio: Audio data
            
        Returns:
            Speaker identifier
        """
        # TODO: Implement actual speaker diarization
        # For now, return a placeholder
        return "unknown"
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        return {
            "state": self.state.value,
            "total_chunks": self.metrics.total_chunks,
            "speech_chunks": self.metrics.speech_chunks,
            "speech_ratio": self.metrics.speech_chunks / max(self.metrics.total_chunks, 1),
            "transcriptions_success": self.metrics.transcriptions_success,
            "transcriptions_failed": self.metrics.transcriptions_failed,
            "transcription_success_rate": (
                self.metrics.transcriptions_success / 
                max(self.metrics.transcriptions_success + self.metrics.transcriptions_failed, 1)
            ),
            "emotion_analyses": self.metrics.emotion_analyses,
            "avg_processing_time": self.metrics.get_avg_processing_time(),
            "queue_size": self.audio_queue.qsize(),
            "last_error": self.metrics.last_error
        }
        
    async def cleanup(self):
        """Cleanup resources."""
        await self.stop_recording()
        self.executor.shutdown(wait=True)


# Global instance (lazy initialization)
_audio_system: Optional[AudioPerceptionSystem] = None


async def get_audio_system() -> AudioPerceptionSystem:
    """Get or create the global audio system instance."""
    global _audio_system
    if _audio_system is None:
        _audio_system = AudioPerceptionSystem()
        await _audio_system.start_recording()
    return _audio_system


async def process_audio_input() -> Optional[Dict[str, Any]]:
    """
    Main interface for reality_bridge to get audio perception data.
    
    Returns:
        Audio perception data or None
    """
    system = await get_audio_system()
    return await system.process_audio()


# Cleanup function for graceful shutdown
async def cleanup_audio_system():
    """Cleanup audio system resources."""
    global _audio_system
    if _audio_system:
        await _audio_system.cleanup()
        _audio_system = None