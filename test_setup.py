"""
Setup Verification Script
Tests that all components are properly installed and configured.
"""

import sys
from pathlib import Path
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stdout, level="INFO", format="<level>{level: <8}</level> | {message}")


def test_imports():
    """Test that all required packages can be imported."""
    print("\n🔍 Testing Package Imports...")
    print("-" * 50)
    
    required_packages = [
        ("langchain", "LangChain core"),
        ("langchain_community", "LangChain community"),
        ("pdfplumber", "PDF processing"),
        ("chromadb", "Vector database"),
        ("ollama", "Ollama client"),
        ("gradio", "Web UI"),
        ("yaml", "YAML config"),
        ("loguru", "Logging")
    ]
    
    all_imported = True
    
    for package, description in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package:20} - {description}")
        except ImportError as e:
            logger.error(f"✗ {package:20} - FAILED: {e}")
            all_imported = False
    
    return all_imported


def test_ollama_connection():
    """Test connection to Ollama service."""
    print("\n🔍 Testing Ollama Connection...")
    print("-" * 50)
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        
        if response.status_code == 200:
            version_data = response.json()
            logger.info(f"✓ Ollama is running (version: {version_data.get('version', 'unknown')})")
            return True
        else:
            logger.error(f"✗ Ollama returned status code: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Cannot connect to Ollama: {e}")
        logger.error("  Make sure Ollama is running: ollama serve")
        return False


def test_ollama_models():
    """Test that required models are downloaded."""
    print("\n🔍 Testing Ollama Models...")
    print("-" * 50)
    
    required_models = [
        "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf",
        "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"
    ]
    
    all_models_present = True
    
    try:
        import ollama
        
        available_models = ollama.list()
        model_names = [model['name'] for model in available_models.get('models', [])]
        
        for model in required_models:
            if any(model in name for name in model_names):
                logger.info(f"✓ {model}")
            else:
                logger.error(f"✗ {model} - NOT FOUND")
                logger.error(f"  Download with: ollama pull {model}")
                all_models_present = False
        
        return all_models_present
        
    except Exception as e:
        logger.error(f"✗ Error checking models: {e}")
        return False


def test_directory_structure():
    """Test that all required directories exist."""
    print("\n🔍 Testing Directory Structure...")
    print("-" * 50)
    
    required_dirs = [
        "modules",
        "config",
        "data",
        "data/uploads",
        "data/vector_db",
        "logs"
    ]
    
    all_dirs_exist = True
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            logger.info(f"✓ {dir_path:20} exists")
        else:
            logger.error(f"✗ {dir_path:20} MISSING")
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"  Created: {dir_path}")
    
    return all_dirs_exist


def test_config_file():
    """Test that config file exists and is valid."""
    print("\n🔍 Testing Configuration File...")
    print("-" * 50)
    
    config_path = Path("config/config.yaml")
    
    if not config_path.exists():
        logger.error(f"✗ config/config.yaml NOT FOUND")
        logger.error("  Please create config file from template")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['models', 'chunking', 'retrieval', 'vector_db', 'upload', 'generation']
        
        for key in required_keys:
            if key in config:
                logger.info(f"✓ Config section '{key}' present")
            else:
                logger.error(f"✗ Config section '{key}' MISSING")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error reading config: {e}")
        return False


def test_module_imports():
    """Test that custom modules can be imported."""
    print("\n🔍 Testing Custom Modules...")
    print("-" * 50)
    
    modules_to_test = [
        "modules.document_loader",
        "modules.text_processor",
        "modules.embedding_engine",
        "modules.vector_store",
        "modules.retrieval_chain"
    ]
    
    all_imported = True
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            logger.info(f"✓ {module_name:35} imported successfully")
        except ImportError as e:
            logger.error(f"✗ {module_name:35} FAILED: {e}")
            all_imported = False
    
    return all_imported


def test_embedding_engine():
    """Test that embedding engine can be initialized."""
    print("\n🔍 Testing Embedding Engine...")
    print("-" * 50)
    
    try:
        from modules.embedding_engine import EmbeddingEngine
        
        # Try to initialize
        engine = EmbeddingEngine()
        logger.info("✓ EmbeddingEngine initialized")
        
        # Try a test embedding
        test_text = "This is a test."
        embedding = engine.embed_query(test_text)
        
        logger.info(f"✓ Test embedding generated (dimension: {len(embedding)})")
        return True
        
    except Exception as e:
        logger.error(f"✗ EmbeddingEngine failed: {e}")
        return False


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("🧠 ResumeBrain Setup Verification")
    print("=" * 60)
    
    results = {
        "Package Imports": test_imports(),
        "Ollama Connection": test_ollama_connection(),
        "Ollama Models": test_ollama_models(),
        "Directory Structure": test_directory_structure(),
        "Config File": test_config_file(),
        "Module Imports": test_module_imports(),
        "Embedding Engine": test_embedding_engine()
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status:8} | {test_name}")
    
    print("\n" + "-" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! You're ready to run ResumeBrain!")
        print("\nTo start the application:")
        print("  python app.py")
        print("\nThen open: http://localhost:7860")
        return True
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Ensure Ollama is running: ollama serve")
        print("  - Download models: ollama pull <model-name>")
        print("  - Install dependencies: pip install -r requirements.txt")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
