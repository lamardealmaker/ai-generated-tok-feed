import pytest
from fastapi.testclient import TestClient
from src.api.routes import app, VideoRequest, PropertyListing, PropertySpecs

@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)

@pytest.fixture
def sample_request():
    """Create sample video generation request."""
    return {
        "listing": {
            "title": "Beautiful Modern Home",
            "price": 750000,
            "description": "Stunning 4-bedroom modern home with amazing views",
            "features": [
                "Open floor plan",
                "Gourmet kitchen",
                "Master suite"
            ],
            "location": "123 Main St, Anytown, USA",
            "specs": {
                "bedrooms": 4,
                "bathrooms": 3.5,
                "square_feet": 3200,
                "lot_size": "0.5 acres",
                "year_built": 2020,
                "property_type": "Single Family"
            }
        },
        "images": [
            "https://example.com/image1.jpg",
            "https://example.com/image2.jpg",
            "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
        ]
    }

def test_generate_video_endpoint(client, sample_request):
    """Test video generation endpoint."""
    response = client.post("/api/v1/videos/generate", json=sample_request)
    assert response.status_code == 200
    
    data = response.json()
    assert "task_id" in data
    assert "estimated_duration" in data
    assert "template_name" in data
    assert "status" in data
    assert data["status"] == "pending"

def test_get_status_endpoint(client, sample_request):
    """Test status endpoint."""
    # First generate a video
    response = client.post("/api/v1/videos/generate", json=sample_request)
    assert response.status_code == 200
    task_id = response.json()["task_id"]
    
    # Then check its status
    response = client.get(f"/api/v1/videos/{task_id}/status")
    assert response.status_code == 200
    
    data = response.json()
    assert data["task_id"] == task_id
    assert "status" in data
    assert "progress" in data
    assert 0 <= data["progress"] <= 100

def test_invalid_request(client):
    """Test invalid request handling."""
    # Missing required fields
    invalid_request = {
        "listing": {
            "title": "Test"
            # Missing price and other required fields
        },
        "images": []
    }
    
    response = client.post("/api/v1/videos/generate", json=invalid_request)
    assert response.status_code == 422  # Validation error

def test_invalid_task_id(client):
    """Test invalid task ID handling."""
    response = client.get("/api/v1/videos/nonexistent-task/status")
    assert response.status_code == 404

def test_validation_price(client, sample_request):
    """Test price validation."""
    request = sample_request.copy()
    request["listing"]["price"] = -100
    
    response = client.post("/api/v1/videos/generate", json=request)
    assert response.status_code == 422

def test_validation_images(client, sample_request):
    """Test image validation."""
    request = sample_request.copy()
    
    # Test empty images
    request["images"] = []
    response = client.post("/api/v1/videos/generate", json=request)
    assert response.status_code == 422
    
    # Test invalid image format
    request["images"] = ["not-a-url-or-base64"]
    response = client.post("/api/v1/videos/generate", json=request)
    assert response.status_code == 422

def test_model_validation():
    """Test Pydantic model validation."""
    # Test valid data
    data = {
        "title": "Test Home",
        "price": 500000,
        "description": "Test description",
        "features": ["Feature 1", "Feature 2"],
        "location": "Test Location",
        "specs": {
            "bedrooms": 3,
            "bathrooms": 2.5
        }
    }
    listing = ListingData(**data)
    assert listing.title == "Test Home"
    assert listing.price == 500000
    
    # Test specs validation
    specs = ListingSpecs(bedrooms=3, bathrooms=2.5)
    assert specs.bedrooms == 3
    assert specs.bathrooms == 2.5
