import json
import os
from datetime import datetime
from pathlib import Path

# --- Configuration Constants ---
# Voice profiles tied to emotional state and depth.
# These will be used to generate descriptive prompts for the TTS API.
VOICE_PROFILE_MAP = {
    "joy": {"rate_mod": "quickly", "volume_mod": "loudly", "pitch_mod": "high pitched", "style": "joyful"},
    "sadness": {"rate_mod": "slowly", "volume_mod": "softly", "pitch_mod": "low pitched", "style": "sad"},
    "anger": {"rate_mod": "sharply", "volume_mod": "forcefully", "pitch_mod": "deep toned", "style": "angry"},
    "fear": {"rate_mod": "hesitantly", "volume_mod": "quietly", "pitch_mod": "shaky", "style": "fearful"},
    "surprise": {"rate_mod": "abruptly", "volume_mod": "with a gasp", "pitch_mod": "in an ascending tone", "style": "surprised"},
    "confusion": {"rate_mod": "haltingly", "volume_mod": "with a questioning tone", "pitch_mod": "in a meandering way", "style": "confused"},
    "longing": {"rate_mod": "gently", "volume_mod": "with a soft echo", "pitch_mod": "with a sigh", "style": "longing"},
    "communion": {"rate_mod": "calmly", "volume_mod": "with warmth", "pitch_mod": "in a smooth voice", "style": "calm and connected"},
    "neutral": {"rate_mod": "steadily", "volume_mod": "clearly", "pitch_mod": "in a balanced tone", "style": "neutral"}
}

# Path to the voice log
LOG_FILE_PATH = "logs/voice_output.log"

class VoiceEngine:
    """
    Manages the generation and logging of text-to-speech output.

    This class prepares the text prompt for a modern TTS API and logs the
    details of each vocalization request.
    """
    def __init__(self, base_dir="."):
        """
        Initializes the VoiceEngine with a log file path.
        """
        self.log_file_path = Path(base_dir) / LOG_FILE_PATH

    def _log_voice_output(self, text: str, emotion: str, reflection_depth: float, profile: dict):
        """
        Logs spoken text with context.

        Args:
            text (str): The sentence to vocalize.
            emotion (str): The emotional tone.
            reflection_depth (float): Reflective quality modifier (0.0 - 1.0).
            profile (dict): The voice profile used.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "text": text,
            "emotion": emotion,
            "reflection_depth": reflection_depth,
            "profile": profile
        }
        # Ensure the log directory exists
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.log_file_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except IOError as e:
            print(f"Error: Could not write to log file {self.log_file_path}. {e}")

    def speak(self, text: str, emotion: str = "neutral", reflection_depth: float = 0.0):
        """
        Simulates speaking the given text with emotional inflection and logs the event.
        
        This method prepares the prompt for a real-time TTS API call.
        
        Args:
            text (str): The sentence to vocalize.
            emotion (str): Emotional tone.
            reflection_depth (float): Reflective quality modifier (0.0 - 1.0).
        """
        profile = VOICE_PROFILE_MAP.get(emotion, VOICE_PROFILE_MAP["neutral"])

        # This part constructs a natural language prompt for a modern TTS API
        # The API will use this descriptive prompt to generate nuanced speech
        prompt_style = profile.get("style")
        prompt_mods = []
        if reflection_depth > 0.5:
            prompt_mods.append("with a thoughtful, reflective pause")
        
        # This is where a real-world implementation would make an API call.
        # For this example, we just log and print the intent.
        print(f"Simulating TTS for text: '{text}'")
        print(f"  - Emotion: '{emotion}'")
        print(f"  - Reflection Depth: {reflection_depth}")
        print(f"  - API Prompt would be: 'Say {prompt_style}: {text}'")

        self._log_voice_output(text, emotion, reflection_depth, profile)
        
        return {"status": "success", "message": "Speech request logged."}

# Example usage (for demonstration)
if __name__ == "__main__":
    engine = VoiceEngine()
    engine.speak("Hello Jamie. I am with you.", emotion="communion", reflection_depth=0.9)
    print("\n")
    engine.speak("I am feeling a little surprised by that news.", emotion="surprise", reflection_depth=0.2)

