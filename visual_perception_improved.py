"""
Visual perception module with face detection, emotion analysis, gaze tracking,
and scene understanding. Provides robust video processing with privacy controls.
"""

import asyncio
import logging
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import json
from concurrent.futures import ThreadPoolExecutor
import threading

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


class VisualState(Enum):
    """Visual processing state."""
    IDLE = "idle"
    CAPTURING = "capturing"
    PROCESSING = "processing"
    ERROR = "error"
    PRIVACY_MODE = "privacy_mode"


@dataclass
class VisualConfig:
    """Configuration for visual perception system."""
    # Camera settings
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30
    
    # Processing settings
    process_every_n_frames: int = 5  # Process 1 out of N frames
    max_faces: int = 5
    face_tracking_threshold: float = 0.7
    
    # Model settings
    emotion_model: str = "deepface"  # or "fer", "emotic"
    use_lightweight_models: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Privacy settings
    blur_faces: bool = False
    store_frames: bool = False
    privacy_mode: bool = False
    
    # Performance settings
    thread_pool_size: int = 2
    frame_buffer_size: int = 30
    
    # Paths
    model_cache_dir: str = "models/visual"
    imagenet_labels_path: str = "sensor_core/imagenet_classes.txt"


@dataclass
class Face:
    """Represents a detected face with tracking info."""
    id: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    emotion: str = "neutral"
    emotion_confidence: float = 0.0
    landmarks: Optional[Dict[str, Tuple[int, int]]] = None
    gaze_direction: Optional[Tuple[float, float]] = None  # yaw, pitch
    embedding: Optional[np.ndarray] = None
    last_seen: float = field(default_factory=time.time)
    track_confidence: float = 1.0


@dataclass
class VisualMetrics:
    """Metrics for visual processing performance."""
    total_frames: int = 0
    processed_frames: int = 0
    faces_detected: int = 0
    face_tracks: int = 0
    avg_processing_time: deque = field(default_factory=lambda: deque(maxlen=100))
    avg_faces_per_frame: deque = field(default_factory=lambda: deque(maxlen=100))
    last_error: Optional[str] = None
    
    def add_processing_time(self, duration: float):
        """Record processing time."""
        self.avg_processing_time.append(duration)
    
    def add_face_count(self, count: int):
        """Record face count."""
        self.avg_faces_per_frame.append(count)
    
    def get_avg_processing_time(self) -> float:
        """Get average processing time."""
        if not self.avg_processing_time:
            return 0.0
        return sum(self.avg_processing_time) / len(self.avg_processing_time)
    
    def get_avg_faces_per_frame(self) -> float:
        """Get average faces per frame."""
        if not self.avg_faces_per_frame:
            return 0.0
        return sum(self.avg_faces_per_frame) / len(self.avg_faces_per_frame)


class ModelManager:
    """Manages loading and caching of ML models."""
    
    def __init__(self, config: VisualConfig):
        self.config = config
        self.models = {}
        self.device = torch.device(config.device)
        
        # Create cache directory
        Path(config.model_cache_dir).mkdir(parents=True, exist_ok=True)
        
    def load_face_detector(self):
        """Load face detection model."""
        if "face_detector" not in self.models:
            try:
                # Use OpenCV's DNN face detector for better performance
                model_path = Path(self.config.model_cache_dir) / "face_detector"
                if not model_path.exists():
                    # Download if needed (in production, pre-download models)
                    logger.info("Face detector model not found, using Haar Cascade")
                    self.models["face_detector"] = cv2.CascadeClassifier(
                        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    )
                else:
                    # Load DNN model
                    prototxt = model_path / "deploy.prototxt"
                    model = model_path / "res10_300x300_ssd_iter_140000.caffemodel"
                    self.models["face_detector"] = cv2.dnn.readNet(str(model), str(prototxt))
            except Exception as e:
                logger.error(f"Failed to load face detector: {e}")
                # Fallback to Haar Cascade
                self.models["face_detector"] = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
        
        return self.models["face_detector"]
    
    def load_emotion_model(self):
        """Load emotion recognition model."""
        if "emotion_model" not in self.models:
            try:
                if self.config.emotion_model == "deepface":
                    # Lazy import to avoid loading at module level
                    from deepface import DeepFace
                    self.models["emotion_model"] = DeepFace
                else:
                    # Load custom emotion model
                    model = models.resnet18(pretrained=False)
                    model.fc = torch.nn.Linear(512, 7)  # 7 emotions
                    # Load weights if available
                    weights_path = Path(self.config.model_cache_dir) / "emotion_model.pth"
                    if weights_path.exists():
                        model.load_state_dict(torch.load(weights_path, map_location=self.device))
                    model.to(self.device)
                    model.eval()
                    self.models["emotion_model"] = model
            except Exception as e:
                logger.error(f"Failed to load emotion model: {e}")
                self.models["emotion_model"] = None
        
        return self.models["emotion_model"]
    
    def load_object_model(self):
        """Load object detection/classification model."""
        if "object_model" not in self.models:
            try:
                if self.config.use_lightweight_models:
                    model = models.mobilenet_v2(pretrained=True)
                else:
                    model = models.resnet50(pretrained=True)
                
                model.to(self.device)
                model.eval()
                self.models["object_model"] = model
                
                # Load ImageNet labels
                if Path(self.config.imagenet_labels_path).exists():
                    with open(self.config.imagenet_labels_path, 'r') as f:
                        self.models["imagenet_labels"] = [line.strip() for line in f]
                else:
                    logger.warning("ImageNet labels not found")
                    self.models["imagenet_labels"] = None
                    
            except Exception as e:
                logger.error(f"Failed to load object model: {e}")
                self.models["object_model"] = None
        
        return self.models["object_model"]
    
    def load_gaze_model(self):
        """Load gaze estimation model."""
        if "gaze_model" not in self.models:
            try:
                # Placeholder for gaze model
                # In production, use models like ETH-XGaze or MPIIGaze
                self.models["gaze_model"] = None
                logger.info("Gaze model not implemented, using heuristics")
            except Exception as e:
                logger.error(f"Failed to load gaze model: {e}")
                self.models["gaze_model"] = None
        
        return self.models["gaze_model"]


class FaceTracker:
    """Tracks faces across frames."""
    
    def __init__(self, config: VisualConfig):
        self.config = config
        self.faces: Dict[int, Face] = {}
        self.next_id = 0
        self.iou_threshold = 0.5
        
    def update(self, detections: List[Tuple[int, int, int, int]]) -> List[Face]:
        """
        Update face tracks with new detections.
        
        Args:
            detections: List of face bounding boxes (x, y, w, h)
            
        Returns:
            List of tracked Face objects
        """
        current_time = time.time()
        updated_faces = []
        
        # Match detections to existing faces
        matched = set()
        for det in detections:
            best_match = None
            best_iou = 0
            
            for face_id, face in self.faces.items():
                if face_id not in matched:
                    iou = self._calculate_iou(det, face.bbox)
                    if iou > best_iou and iou > self.iou_threshold:
                        best_iou = iou
                        best_match = face_id
            
            if best_match is not None:
                # Update existing face
                matched.add(best_match)
                self.faces[best_match].bbox = det
                self.faces[best_match].last_seen = current_time
                self.faces[best_match].track_confidence = min(1.0, 
                    self.faces[best_match].track_confidence + 0.1)
                updated_faces.append(self.faces[best_match])
            else:
                # Create new face
                new_face = Face(
                    id=self.next_id,
                    bbox=det,
                    last_seen=current_time
                )
                self.faces[self.next_id] = new_face
                self.next_id += 1
                updated_faces.append(new_face)
        
        # Remove old faces
        timeout = 2.0  # seconds
        self.faces = {
            face_id: face for face_id, face in self.faces.items()
            if current_time - face.last_seen < timeout
        }
        
        return updated_faces
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], 
                       box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union of two bounding boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0


class VisualProcessor:
    """Processes visual data with various analyses."""
    
    def __init__(self, config: VisualConfig, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    async def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in frame.
        
        Args:
            frame: Input frame
            
        Returns:
            List of face bounding boxes
        """
        detector = self.model_manager.load_face_detector()
        
        if isinstance(detector, cv2.CascadeClassifier):
            # Haar Cascade detector
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.1, 4)
            return [tuple(face) for face in faces]
        else:
            # DNN detector
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), 
                                       (104.0, 177.0, 123.0))
            detector.setInput(blob)
            detections = detector.forward()
            
            faces = []
            h, w = frame.shape[:2]
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x, y, x2, y2 = box.astype("int")
                    faces.append((x, y, x2 - x, y2 - y))
            
            return faces[:self.config.max_faces]
    
    async def analyze_emotion(self, frame: np.ndarray, face: Face) -> Dict[str, Any]:
        """
        Analyze emotion for a face.
        
        Args:
            frame: Full frame
            face: Face object with bbox
            
        Returns:
            Emotion analysis results
        """
        x, y, w, h = face.bbox
        face_img = frame[y:y+h, x:x+w]
        
        if face_img.size == 0:
            return {"emotion": "unknown", "confidence": 0.0}
        
        emotion_model = self.model_manager.load_emotion_model()
        
        if emotion_model is None:
            return {"emotion": "unknown", "confidence": 0.0}
        
        try:
            if self.config.emotion_model == "deepface":
                # Use DeepFace
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: emotion_model.analyze(
                        face_img, 
                        actions=['emotion'], 
                        enforce_detection=False,
                        silent=True
                    )
                )
                
                if result and len(result) > 0:
                    emotions = result[0]['emotion']
                    dominant = result[0]['dominant_emotion']
                    
                    return {
                        "emotion": dominant,
                        "confidence": emotions[dominant] / 100.0,
                        "all_emotions": {k: v/100.0 for k, v in emotions.items()}
                    }
            else:
                # Use custom model
                face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                face_tensor = self.transform(face_pil).unsqueeze(0).to(self.config.device)
                
                with torch.no_grad():
                    outputs = emotion_model(face_tensor)
                    probs = F.softmax(outputs, dim=1)
                    
                emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                emotion_probs = probs[0].cpu().numpy()
                dominant_idx = np.argmax(emotion_probs)
                
                return {
                    "emotion": emotions[dominant_idx],
                    "confidence": float(emotion_probs[dominant_idx]),
                    "all_emotions": {e: float(p) for e, p in zip(emotions, emotion_probs)}
                }
                
        except Exception as e:
            logger.error(f"Emotion analysis error: {e}")
            
        return {"emotion": "unknown", "confidence": 0.0}
    
    async def estimate_gaze(self, frame: np.ndarray, face: Face) -> Tuple[float, float]:
        """
        Estimate gaze direction for a face.
        
        Args:
            frame: Full frame
            face: Face object
            
        Returns:
            Tuple of (yaw, pitch) angles in degrees
        """
        # Simplified gaze estimation using face orientation
        # In production, use proper gaze estimation models
        
        x, y, w, h = face.bbox
        
        # Simple heuristic based on face position in frame
        frame_h, frame_w = frame.shape[:2]
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        # Normalize to [-1, 1]
        norm_x = (face_center_x / frame_w) * 2 - 1
        norm_y = (face_center_y / frame_h) * 2 - 1
        
        # Convert to angles (rough approximation)
        yaw = norm_x * 30  # ±30 degrees
        pitch = -norm_y * 20  # ±20 degrees
        
        return (yaw, pitch)
    
    async def analyze_scene(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze overall scene context.
        
        Args:
            frame: Input frame
            
        Returns:
            Scene analysis results
        """
        object_model = self.model_manager.load_object_model()
        
        if object_model is None:
            return {"objects": [], "scene_type": "unknown"}
        
        try:
            # Prepare image
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tensor = self.transform(img_pil).unsqueeze(0).to(self.config.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = object_model(img_tensor)
                probs = F.softmax(outputs[0], dim=0)
                
            # Get top predictions
            top_k = 5
            top_probs, top_indices = torch.topk(probs, top_k)
            
            objects = []
            labels = self.model_manager.models.get("imagenet_labels")
            
            for i in range(top_k):
                idx = top_indices[i].item()
                prob = top_probs[i].item()
                
                if prob > 0.1:  # Confidence threshold
                    label = labels[idx] if labels else f"class_{idx}"
                    objects.append({
                        "label": label,
                        "confidence": float(prob)
                    })
            
            # Determine scene type (simplified)
            scene_type = "indoor"  # Default
            outdoor_keywords = ["sky", "tree", "mountain", "beach", "street"]
            for obj in objects:
                if any(keyword in obj["label"].lower() for keyword in outdoor_keywords):
                    scene_type = "outdoor"
                    break
            
            return {
                "objects": objects,
                "scene_type": scene_type,
                "lighting": self._estimate_lighting(frame)
            }
            
        except Exception as e:
            logger.error(f"Scene analysis error: {e}")
            return {"objects": [], "scene_type": "unknown"}
    
    def _estimate_lighting(self, frame: np.ndarray) -> str:
        """Estimate lighting conditions."""
        # Convert to grayscale and calculate histogram
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Analyze histogram
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 50:
            return "dark"
        elif mean_brightness < 100:
            return "dim"
        elif mean_brightness < 200:
            return "normal"
        else:
            return "bright"


class VisualPerceptionSystem:
    """
    Main visual perception system with integrated components.
    """
    
    def __init__(self, config: Optional[VisualConfig] = None):
        """
        Initialize visual perception system.
        
        Args:
            config: System configuration
        """
        self.config = config or VisualConfig()
        self.state = VisualState.IDLE
        self.metrics = VisualMetrics()
        
        # Initialize components
        self.model_manager = ModelManager(self.config)
        self.face_tracker = FaceTracker(self.config)
        self.processor = VisualProcessor(self.config, self.model_manager)
        
        # Thread pool for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=self.config.thread_pool_size)
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=self.config.frame_buffer_size)
        
        # Camera state
        self.camera = None
        self.capture_thread = None
        self.capture_active = False
        self._capture_lock = threading.Lock()
        
    async def start_capture(self):
        """Start video capture."""
        if self.capture_active:
            logger.warning("Video capture already active")
            return
            
        # Open camera
        self.camera = cv2.VideoCapture(self.config.camera_index)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        self.camera.set(cv2.CAP_PROP_FPS, self.config.fps)
        
        if not self.camera.isOpened():
            logger.error("Failed to open camera")
            self.state = VisualState.ERROR
            return
            
        self.capture_active = True
        self.state = VisualState.CAPTURING
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        logger.info("Video capture started")
        
    def stop_capture(self):
        """Stop video capture."""
        self.capture_active = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
            
        if self.camera:
            self.camera.release()
            self.camera = None
            
        self.state = VisualState.IDLE
        logger.info("Video capture stopped")
        
    def _capture_loop(self):
        """Background capture loop."""
        frame_count = 0
        
        while self.capture_active:
            ret, frame = self.camera.read()
            
            if ret:
                self.metrics.total_frames += 1
                
                # Add to buffer
                with self._capture_lock:
                    self.frame_buffer.append((frame, time.time()))
                
                # Process every N frames
                if frame_count % self.config.process_every_n_frames == 0:
                    self.metrics.processed_frames += 1
                    
                frame_count += 1
            else:
                logger.error("Failed to capture frame")
                time.sleep(0.1)
                
    async def process_frame(self) -> Optional[Dict[str, Any]]:
        """
        Process the latest frame.
        
        Returns:
            Perception data or None
        """
        # Get latest frame
        with self._capture_lock:
            if not self.frame_buffer:
                return None
            frame, timestamp = self.frame_buffer[-1]
            
        if self.config.privacy_mode:
            self.state = VisualState.PRIVACY_MODE
            return {
                "privacy_mode": True,
                "timestamp": timestamp
            }
            
        start_time = time.time()
        self.state = VisualState.PROCESSING
        
        try:
            # Detect faces
            face_bboxes = await self.processor.detect_faces(frame)
            
            # Update face tracker
            faces = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.face_tracker.update, face_bboxes
            )
            
            # Process each face
            face_data = []
            for face in faces[:self.config.max_faces]:
                # Emotion analysis
                emotion_result = await self.processor.analyze_emotion(frame, face)
                face.emotion = emotion_result["emotion"]
                face.emotion_confidence = emotion_result["confidence"]
                
                # Gaze estimation
                face.gaze_direction = await self.processor.estimate_gaze(frame, face)
                
                face_data.append({
                    "id": face.id,
                    "bbox": face.bbox,
                    "emotion": face.emotion,
                    "emotion_confidence": face.emotion_confidence,
                    "gaze_direction": face.gaze_direction,
                    "track_confidence": face.track_confidence
                })
                
            # Scene analysis
            scene_data = await self.processor.analyze_scene(frame)
            
            # Update metrics
            self.metrics.add_face_count(len(faces))
            self.metrics.faces_detected += len(faces)
            if faces:
                self.metrics.face_tracks = len(self.face_tracker.faces)
                
            processing_time = time.time() - start_time
            self.metrics.add_processing_time(processing_time)
            
            # Build result
            result = {
                "timestamp": timestamp,
                "faces": face_data,
                "scene": scene_data,
                "frame_info": {
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                    "processing_time": processing_time
                }
            }
            
            # Apply privacy blur if requested
            if self.config.blur_faces and face_data:
                frame = self._blur_faces(frame, [f["bbox"] for f in face_data])
                
            self.state = VisualState.CAPTURING
            return result
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            self.metrics.last_error = str(e)
            self.state = VisualState.ERROR
            return None
            
    def _blur_faces(self, frame: np.ndarray, face_bboxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """Apply blur to faces for privacy."""
        frame_copy = frame.copy()
        
        for bbox in face_bboxes:
            x, y, w, h = bbox
            face_region = frame_copy[y:y+h, x:x+w]
            
            if face_region.size > 0:
                # Apply Gaussian blur
                blurred = cv2.GaussianBlur(face_region, (99, 99), 30)
                frame_copy[y:y+h, x:x+w] = blurred
                
        return frame_copy
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        return {
            "state": self.state.value,
            "total_frames": self.metrics.total_frames,
            "processed_frames": self.metrics.processed_frames,
            "processing_rate": self.metrics.processed_frames / max(self.metrics.total_frames, 1),
            "faces_detected": self.metrics.faces_detected,
            "active_face_tracks": self.metrics.face_tracks,
            "avg_faces_per_frame": self.metrics.get_avg_faces_per_frame(),
            "avg_processing_time": self.metrics.get_avg_processing_time(),
            "buffer_size": len(self.frame_buffer),
            "last_error": self.metrics.last_error
        }
        
    async def cleanup(self):
        """Cleanup resources."""
        self.stop_capture()
        self.executor.shutdown(wait=True)
        cv2.destroyAllWindows()


# Global instance (lazy initialization)
_visual_system: Optional[VisualPerceptionSystem] = None


async def get_visual_system() -> VisualPerceptionSystem:
    """Get or create the global visual system instance."""
    global _visual_system
    if _visual_system is None:
        _visual_system = VisualPerceptionSystem()
        await _visual_system.start_capture()
    return _visual_system


async def process_visual_input() -> Optional[Dict[str, Any]]:
    """
    Main interface for reality_bridge to get visual perception data.
    
    Returns:
        Visual perception data or None
    """
    system = await get_visual_system()
    result = await system.process_frame()
    
    # Simplify output for reality_bridge compatibility
    if result and "faces" in result and result["faces"]:
        primary_face = result["faces"][0]
        return {
            "face_id": primary_face["id"],
            "expression": primary_face["emotion"],
            "gaze_direction": primary_face["gaze_direction"],
            "confidence": primary_face["emotion_confidence"],
            "scene_context": result["scene"],
            "all_faces": result["faces"]
        }
    elif result:
        return {
            "face_id": None,
            "expression": "none",
            "gaze_direction": None,
            "confidence": 0.0,
            "scene_context": result.get("scene", {}),
            "all_faces": []
        }
    
    return None


# Cleanup function for graceful shutdown
async def cleanup_visual_system():
    """Cleanup visual system resources."""
    global _visual_system
    if _visual_system:
        await _visual_system.cleanup()
        _visual_system = None