#!/bin/bash

# ResumeBrain Setup Script
# Automates project setup and model downloads

set -e  # Exit on error

echo "🧠 ResumeBrain Setup Script"
echo "================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check if Ollama is installed
echo "Step 1: Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    print_success "Ollama is installed"
    ollama --version
else
    print_error "Ollama is not installed"
    echo ""
    echo "Please install Ollama from: https://ollama.com"
    echo ""
    echo "macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh"
    echo "Windows: Download from https://ollama.com"
    exit 1
fi

echo ""

# Check if Ollama is running
echo "Step 2: Checking Ollama service..."
if curl -s http://localhost:11434/api/version > /dev/null; then
    print_success "Ollama service is running"
else
    print_info "Ollama service is not running. Starting it..."
    ollama serve &
    sleep 3
    
    if curl -s http://localhost:11434/api/version > /dev/null; then
        print_success "Ollama service started"
    else
        print_error "Failed to start Ollama service"
        echo "Please start Ollama manually: ollama serve"
        exit 1
    fi
fi

echo ""

# Download embedding model
echo "Step 3: Downloading embedding model..."
EMBEDDING_MODEL="hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"

if ollama list | grep -q "$EMBEDDING_MODEL"; then
    print_success "Embedding model already downloaded"
else
    print_info "Downloading $EMBEDDING_MODEL..."
    ollama pull "$EMBEDDING_MODEL"
    print_success "Embedding model downloaded"
fi

echo ""

# Download LLM model
echo "Step 4: Downloading LLM model..."
LLM_MODEL="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

if ollama list | grep -q "$LLM_MODEL"; then
    print_success "LLM model already downloaded"
else
    print_info "Downloading $LLM_MODEL (this may take a few minutes)..."
    ollama pull "$LLM_MODEL"
    print_success "LLM model downloaded"
fi

echo ""

# Create project directories
echo "Step 5: Creating project directories..."
mkdir -p modules config data/uploads data/vector_db logs
print_success "Directories created"

echo ""

# Check Python version
echo "Step 6: Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Python $PYTHON_VERSION installed"
else
    print_error "Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo ""

# Create virtual environment
echo "Step 7: Setting up Python virtual environment..."
if [ -d "venv" ]; then
    print_info "Virtual environment already exists"
else
    python3 -m venv venv
    print_success "Virtual environment created"
fi

echo ""

# Install dependencies using virtual environment's pip directly
echo "Step 8: Installing Python dependencies..."
print_info "Installing packages using virtual environment..."

# Detect OS for pip path
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    PIP_CMD="venv/Scripts/pip"
else
    # Try pip3 first, fallback to pip
    if [ -f "venv/bin/pip3" ]; then
        PIP_CMD="venv/bin/pip3"
    else
        PIP_CMD="venv/bin/pip"
    fi
fi

# Check if pip exists
if [ ! -f "$PIP_CMD" ]; then
    print_error "pip not found in virtual environment"
    print_info "Recreating virtual environment..."
    rm -rf venv
    python3 -m venv venv
    if [ -f "venv/bin/pip3" ]; then
        PIP_CMD="venv/bin/pip3"
    else
        PIP_CMD="venv/bin/pip"
    fi
fi

$PIP_CMD install --upgrade pip > /dev/null 2>&1
$PIP_CMD install -r requirements.txt

if [ $? -eq 0 ]; then
    print_success "Python dependencies installed"
else
    print_error "Failed to install dependencies"
    exit 1
fi

echo ""

# Verify installation
echo "Step 9: Verifying installation..."

# Detect Python command for verification
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    PYTHON_CMD="venv/Scripts/python"
else
    # Try python3 first, fallback to python
    if [ -f "venv/bin/python3" ]; then
        PYTHON_CMD="venv/bin/python3"
    else
        PYTHON_CMD="venv/bin/python"
    fi
fi

# Check if key packages are installed
$PYTHON_CMD -c "import langchain, gradio, chromadb, ollama" 2>/dev/null

if [ $? -eq 0 ]; then
    print_success "All packages verified"
else
    print_error "Some packages failed to install"
    exit 1
fi

echo ""

# Summary
echo "================================"
echo "🎉 Setup Complete!"
echo "================================"
echo ""
echo "✓ Ollama installed and running"
echo "✓ Models downloaded:"
echo "  - $EMBEDDING_MODEL"
echo "  - $LLM_MODEL"
echo "✓ Project directories created"
echo "✓ Python dependencies installed"
echo ""
echo "To start the application:"
echo ""
echo "  Option 1: Activate virtual environment (if activate exists):"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "     venv\\Scripts\\activate"
else
    echo "     source venv/bin/activate"
fi
echo "     python app_main.py"
echo ""
echo "  Option 2: Use virtual environment directly:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "     venv\\Scripts\\python app_main.py"
else
    echo "     venv/bin/python3 app_main.py"
fi
echo ""
echo "  3. Open browser to: http://localhost:7860"
echo ""
echo "Happy RAG building! 🚀"
