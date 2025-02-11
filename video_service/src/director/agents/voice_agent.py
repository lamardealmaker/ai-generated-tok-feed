from typing import Dict, Any, Optional
import os
import json
import aiohttp
from pathlib import Path

class VoiceAgent:
    """
    Agent for generating voiceovers using ElevenLabs API.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.api_key = os.getenv('ELEVENLABS_API_KEY')
        self.base_url = 'https://api.elevenlabs.io/v1'
        
    def _default_config(self) -> Dict[str, Any]:
        """Default voice configuration."""
        return {
            'voices': {
                'professional': {
                    'voice_id': 'ErXwobaYiN019PkySvjV',  # Antoni voice
                    'settings': {
                        'stability': 0.75,
                        'similarity_boost': 0.75
                    }
                },
                'friendly': {
                    'voice_id': 'MF3mGyEYCl7XYWbV9V6O',  # Elli voice
                    'settings': {
                        'stability': 0.65,
                        'similarity_boost': 0.85
                    }
                },
                'luxury': {
                    'voice_id': 'VR6AewLTigWG4xSOukaG',  # Adam voice
                    'settings': {
                        'stability': 0.85,
                        'similarity_boost': 0.65
                    }
                }
            },
            'default_voice': 'professional'
        }
        
    async def generate_voiceover(self,
                               script: str,
                               style: str = 'professional',
                               output_path: Optional[str] = None) -> str:
        """
        Generate voiceover for script using ElevenLabs API.
        
        Args:
            script: Text to convert to speech
            style: Voice style to use
            output_path: Optional path to save audio file
            
        Returns:
            Path to generated audio file
        """
        if not self.api_key:
            raise ValueError("ElevenLabs API key not found in environment variables")
            
        voice_config = self.config['voices'].get(style, 
                                               self.config['voices'][self.config['default_voice']])
                                               
        headers = {
            'xi-api-key': self.api_key,
            'Content-Type': 'application/json'
        }
        
        data = {
            'text': script,
            'voice_settings': voice_config['settings']
        }
        
        url = f"{self.base_url}/text-to-speech/{voice_config['voice_id']}"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status != 200:
                    raise Exception(f"ElevenLabs API error: {await response.text()}")
                    
                audio_data = await response.read()
                
        if not output_path:
            output_path = f"temp_voiceover_{style}_{hash(script)}.mp3"
            
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(audio_data)
            
        return str(output_path)
        
    async def get_voice_info(self, voice_id: str) -> Dict[str, Any]:
        """Get information about a specific voice."""
        if not self.api_key:
            raise ValueError("ElevenLabs API key not found in environment variables")
            
        headers = {'xi-api-key': self.api_key}
        url = f"{self.base_url}/voices/{voice_id}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    raise Exception(f"ElevenLabs API error: {await response.text()}")
                return await response.json()
                
    def get_voice_settings(self, style: str) -> Dict[str, Any]:
        """Get voice settings for a specific style."""
        return self.config['voices'].get(style, 
                                      self.config['voices'][self.config['default_voice']])
