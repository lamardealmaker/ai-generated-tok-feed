from typing import Dict, Any, Optional
import os
import logging
import random
from ..core.base_tool import BaseTool

logger = logging.getLogger(__name__)

class VoiceoverTool(BaseTool):
    """Tool for generating voiceovers using ElevenLabs."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        super().__init__(config)
        if not self.api_key:
            logger.warning("ELEVENLABS_API_KEY not found in environment")
        
        self.model = os.getenv("ELEVENLABS_MODEL", "eleven_monolingual_v1")
        self.default_stability = float(os.getenv("ELEVENLABS_VOICE_STABILITY", "0.75"))
        self.default_similarity = float(os.getenv("ELEVENLABS_VOICE_SIMILARITY", "0.75"))
            
        self.voice_profiles = {
            "luxury": {
                "voice_id": "ErXwobaYiN019PkySvjV",
                "settings": {
                    "stability": self.default_stability,
                    "similarity_boost": self.default_similarity,
                    "style": 0.5,
                    "use_speaker_boost": True
                }
            },
            "friendly": {
                "voice_id": "EXAVITQu4vr4xnSDxMaL",
                "settings": {
                    "stability": self.default_stability,
                    "similarity_boost": self.default_similarity,
                    "style": 0.3,
                    "use_speaker_boost": True
                }
            },
            "professional": {
                "voice_id": "VR6AewLTigWG4xSOukaG",
                "settings": {
                    "stability": self.default_stability,
                    "similarity_boost": self.default_similarity,
                    "style": 0.4,
                    "use_speaker_boost": True
                }
            }
        }
        
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "voiceover_tool",
            "description": "Generates professional voiceovers using ElevenLabs API",
            "requires_api_key": True,
            "supported_voices": list(self.voice_profiles.keys()),
            "max_text_length": 5000,
            "supported_formats": ["mp3", "wav"]
        }
        
    def generate_voiceover(
        self,
        text: str,
        style: str = "professional",
        output_format: str = "mp3"
    ) -> Dict[str, Any]:
        """
        Generate a voiceover for the given text using ElevenLabs API.
        
        This method ignores the passed style and randomly selects one of the default voice profiles.
        
        Args:
            text: Text to convert to speech
            style: (Ignored) Voice style
            output_format: Output audio format
            
        Returns:
            Dictionary containing voiceover metadata and file path
        """
        if not self.api_key:
            raise ValueError("ElevenLabs API key not configured")
            
        # Randomly select a voice from the default profiles
        selected_style = random.choice(list(self.voice_profiles.keys()))
        profile = self.voice_profiles[selected_style]
            
        if len(text) > self.get_capabilities()["max_text_length"]:
            raise ValueError("Text exceeds maximum length")
            
        try:
            import requests
            import uuid
            
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{profile['voice_id']}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            
            data = {
                "text": text,
                "model_id": self.model,
                "voice_settings": profile["settings"]
            }
            
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                output_dir = os.path.join(os.getcwd(), "assets/audio/voiceover")
                os.makedirs(output_dir, exist_ok=True)
                
                file_name = f"voiceover_{uuid.uuid4()}.{output_format}"
                file_path = os.path.join(output_dir, file_name)
                
                with open(file_path, "wb") as f:
                    f.write(response.content)
                
                return {
                    "status": "success",
                    "file_path": file_path,
                    "text": text,
                    "voice_id": profile["voice_id"],
                    "settings": profile["settings"],
                    "selected_voice": selected_style
                }
            else:
                error_msg = f"ElevenLabs API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
        except Exception as e:
            logger.error(f"Voiceover generation failed: {str(e)}")
            raise
            
    def validate_config(self) -> None:
        if not self.api_key and not self.config.get("mock_mode", False):
            logger.warning("No API key provided and mock mode is disabled")