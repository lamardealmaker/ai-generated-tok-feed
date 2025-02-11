import asyncio
import os
from dotenv import load_dotenv
from src.director.core.reasoning_engine import ReasoningEngine
from src.utils.logging import setup_logger

# Load environment variables
load_dotenv()

# Set up logger
logger = setup_logger("test_video_gen")

async def test_video_generation():
    try:
        # Initialize the engine
        engine = ReasoningEngine()
        
        # Firebase image URLs (using fewer images for faster testing)
        image_urls = [
            "gs://project3-ziltok.firebasestorage.app/test_properties/house1.jpg",
            "gs://project3-ziltok.firebasestorage.app/test_properties/interior1.jpg",
            "gs://project3-ziltok.firebasestorage.app/test_properties/backyard.jpg"
        ]
        
        # Sample property data
        listing_data = {
            "id": "test-luxury-home-1",
            "title": "Luxurious Modern Estate",
            "price": 1250000,
            "description": "Stunning modern estate with breathtaking views. This masterpiece of design features an open concept living space, gourmet kitchen, and resort-style backyard.",
            "features": [
                "Gourmet Kitchen with High-End Appliances",
                "Open Concept Floor Plan",
                "Resort-Style Backyard with Pool",
                "Smart Home Technology",
                "Hardwood Floors Throughout"
            ],
            "location": "Beverly Hills",
            "bedrooms": 5,
            "bathrooms": 4.5,
            "sqft": 4200,
            "year_built": 2022,
            "lot_size": 12000
        }
        
        # Define progress callback
        def progress_callback(progress: float, message: str):
            logger.info(f"Progress: {progress*100:.1f}% - {message}")
        
        logger.info("Starting video generation test...")
        
        # Generate video
        task = await engine.generate_video(
            listing_data=listing_data,
            image_urls=image_urls,
            style="luxury",
            duration=15.0,  # Shorter duration for testing
            progress_callback=progress_callback
        )
        
        logger.info(f"Video generation task created with ID: {task.id}")
        
        # Wait for completion
        while task.status == "processing":
            logger.info(f"Task progress: {task.progress*100:.1f}%")
            await asyncio.sleep(2)
        
        if task.status == "completed":
            logger.info("Video generation completed successfully!")
            logger.info(f"Result: {task.result}")
        else:
            logger.error(f"Video generation failed: {task.error}")
            
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(test_video_generation())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
