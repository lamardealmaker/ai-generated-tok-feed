import asyncio
import os
from dotenv import load_dotenv
from src.director.core.reasoning_engine import ReasoningEngine
from src.utils.logging import setup_logger

# Load environment variables
load_dotenv()

# Set up logger
logger = setup_logger("property_video_gen")

async def generate_property_video(property_id: str, style: str = "luxury"):
    try:
        # Initialize the engine
        engine = ReasoningEngine()
        
        # Real property data
        listing_data = {
            "id": property_id,
            "title": "Modern Waterfront Paradise",
            "price": 2850000,
            "description": """
            Discover luxury living at its finest in this stunning waterfront estate. 
            This architectural masterpiece seamlessly blends indoor and outdoor living, 
            offering breathtaking water views from nearly every room. The home features 
            soaring ceilings, walls of glass, and premium finishes throughout.
            """,
            "features": [
                "Panoramic Water Views",
                "Chef's Kitchen with Premium Appliances",
                "Primary Suite with Private Terrace",
                "Indoor-Outdoor Entertainment Spaces",
                "Private Dock",
                "Wine Cellar",
                "Smart Home Technology",
                "3-Car Garage"
            ],
            "location": "Miami Beach",
            "bedrooms": 6,
            "bathrooms": 7.5,
            "sqft": 6800,
            "year_built": 2023,
            "lot_size": 15000
        }
        
        # Property images (using test images for now)
        image_urls = [
            "gs://project3-ziltok.firebasestorage.app/test_properties/house1.jpg",
            "gs://project3-ziltok.firebasestorage.app/test_properties/house2.jpg",
            "gs://project3-ziltok.firebasestorage.app/test_properties/house3.jpg",
            "gs://project3-ziltok.firebasestorage.app/test_properties/interior1.jpg",
            "gs://project3-ziltok.firebasestorage.app/test_properties/interior2.jpg",
            "gs://project3-ziltok.firebasestorage.app/test_properties/backyard.jpg"
        ]
        
        # Progress callback
        def progress_callback(progress: float, message: str):
            logger.info(f"Progress: {progress*100:.1f}% - {message}")
        
        logger.info(f"Generating video for property {property_id}...")
        
        # Generate video
        await engine.generate_video(
            listing_data=listing_data,
            image_urls=image_urls,
            style=style,
            duration=30.0,  # 30 second video
            progress_callback=progress_callback
        )
        
        logger.info("Video generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Video generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # You can change the style to "modern" or "minimal" for different looks
        asyncio.run(generate_property_video("waterfront-estate-1", style="luxury"))
    except KeyboardInterrupt:
        logger.info("Video generation interrupted by user")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
