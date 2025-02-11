import pytest
from pathlib import Path
import yaml
from src.director.core.engine import ReasoningEngine
from src.director.core.session import SessionManager, Session
from src.director.core.config import DirectorConfig

@pytest.fixture
def config_dir(tmp_path):
    return tmp_path / "config" / "director"

@pytest.fixture
def engine(config_dir):
    config_dir.mkdir(parents=True)
    return ReasoningEngine(str(config_dir / "engine.yaml"))

@pytest.fixture
def session_manager():
    return SessionManager()

def test_engine_initialization(engine):
    """Test that the engine initializes with default config."""
    assert engine.config is not None
    assert 'llm' in engine.config
    assert 'video' in engine.config
    assert 'agents' in engine.config

def test_session_creation(session_manager):
    """Test session creation and retrieval."""
    session = session_manager.create_session()
    assert session.session_id is not None
    assert session.status == "initialized"
    assert session.progress == 0
    
    # Test retrieval
    retrieved = session_manager.get_session(session.session_id)
    assert retrieved is not None
    assert retrieved.session_id == session.session_id

def test_session_updates(session_manager):
    """Test session progress updates."""
    session = session_manager.create_session()
    session_manager.update_session(session.session_id, 50, "processing")
    
    updated = session_manager.get_session(session.session_id)
    assert updated.progress == 50
    assert updated.status == "processing"

def test_config_management(config_dir):
    """Test configuration management."""
    config = DirectorConfig(str(config_dir))
    
    # Test saving config
    test_config = {
        'test_key': 'test_value'
    }
    config.save_config('test', test_config)
    
    # Test loading config
    loaded = config.load_config('test')
    assert loaded['test_key'] == 'test_value'
    
    # Test getting specific value
    value = config.get_value('test', 'test_key')
    assert value == 'test_value'

@pytest.mark.asyncio
async def test_engine_request_processing(engine, session_manager):
    """Test processing a video generation request."""
    engine.initialize(session_manager)
    session = session_manager.create_session()
    
    request_data = {
        "listing": {
            "title": "Test Property",
            "price": 500000,
            "description": "Beautiful test property"
        },
        "images": ["test1.jpg", "test2.jpg"]
    }
    
    result = await engine.process_request(session.session_id, request_data)
    assert result["status"] == "processing"
    assert result["session_id"] == session.session_id
