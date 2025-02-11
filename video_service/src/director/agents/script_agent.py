import openai
from typing import Dict, Any
from ..core.base_agent import BaseAgent

class ScriptAgent(BaseAgent):
    """
    Agent for generating a cohesive video script using OpenAI.
    
    This implementation uses detailed property information (title, location, price, square footage,
    description, features, etc.) to generate a single, cohesive and factual script. The script is generated
    all at once and is intended to be passed to the voiceover tool, ensuring that the narration matches the
    property details without hyperbole or repetitive content.
    """
    
    def __init__(self):
        # No template-based generation; we rely solely on OpenAI for script generation.
        pass

    def generate_script(self, property_data: Dict[str, Any], style: str = "modern", duration: float = 30.0) -> Dict[str, Any]:
        """
        Generate a cohesive video script for a property using OpenAI's API.
        
        Args:
            property_data: Dictionary containing detailed property information (title, location, price, sqft,
                           bedrooms, bathrooms, description, features, etc.)
            style: Video style to use (informational; this parameter is not used in script generation)
            duration: Target video duration in seconds
        
        Returns:
            Dictionary containing one script segment that covers the full duration of the video.
        """
        # Construct a prompt using the property details to ensure a cohesive, factual script.
        prompt = (
            "Generate a clear, factual, and cohesive video script for a real estate property with the following details:\n\n"
            f"Title: {property_data.get('title', 'No Title')}\n"
            f"Location: {property_data.get('location', 'No Location')}\n"
            f"Price: ${property_data.get('price', 'N/A')}\n"
            f"Square Footage: {property_data.get('sqft', 'N/A')} sqft\n"
            f"Bedrooms: {property_data.get('bedrooms', 'N/A')}\n"
            f"Bathrooms: {property_data.get('bathrooms', 'N/A')}\n"
            f"Lot Size: {property_data.get('lot_size', 'N/A')} sqft\n"
            f"Year Built: {property_data.get('year_built', 'N/A')}\n"
            f"Description: {property_data.get('description', '')}\n"
            f"Features: {', '.join(property_data.get('features', []))}\n\n"
            "Please generate a video script that is factual, clear, and concise. The script should cover all the above information "
            "in a coherent manner, without using hyperbolic language or repetitive phrases. It should be suitable for a voiceover narration "
            "in a real estate video and should last for the entire duration specified."
        )

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional real estate script writer who generates factual and engaging scripts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        
        script_text = response.choices[0].message.content.strip()
        
        # Return the cohesive script as one segment covering the entire video duration.
        return {
            "style": style,
            "duration": duration,
            "segments": [
                {
                    "type": "full_script",
                    "text": script_text,
                    "start_time": 0.0,
                    "duration": duration,
                    "focus": "property"
                }
            ],
            "metadata": property_data
        }
