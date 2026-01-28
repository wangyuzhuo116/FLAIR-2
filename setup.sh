#!/bin/bash

# FLAIR-2 Setup Script
# This script helps set up the environment and verify the installation

set -e  # Exit on error

echo "=========================================="
echo "FLAIR-2 Dual-Stream Model Setup"
echo "=========================================="

# Check Python version
echo ""
echo "1. Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "   Python version: $python_version"

# Check if Python >= 3.8
required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "   ✗ Python 3.8 or higher required"
    exit 1
else
    echo "   ✓ Python version OK"
fi

# Install dependencies
echo ""
echo "2. Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
    echo "   ✓ Dependencies installed"
else
    echo "   ✗ requirements.txt not found"
    exit 1
fi

# Create necessary directories
echo ""
echo "3. Creating directories..."
mkdir -p logs
mkdir -p checkpoints
mkdir -p predictions
mkdir -p visualizations
mkdir -p data_splits
echo "   ✓ Directories created"

# Verify installation
echo ""
echo "4. Verifying installation..."
if python test_implementation.py > /dev/null 2>&1; then
    echo "   ✓ Installation verified successfully"
else
    echo "   ✗ Verification failed. Running detailed test..."
    python test_implementation.py
    exit 1
fi

# Run example
echo ""
echo "5. Running example..."
if python example.py > /dev/null 2>&1; then
    echo "   ✓ Example ran successfully"
else
    echo "   ✗ Example failed"
    exit 1
fi

# Check for CUDA
echo ""
echo "6. Checking CUDA availability..."
cuda_available=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [ "$cuda_available" = "True" ]; then
    cuda_version=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
    gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    echo "   ✓ CUDA available"
    echo "     CUDA version: $cuda_version"
    echo "     GPU: $gpu_name"
else
    echo "   ⚠ CUDA not available (CPU mode)"
    echo "     Training will be slow without GPU"
fi

# Summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Prepare your FLAIR-2 dataset:"
echo "   - Download from: https://ignf.github.io/FLAIR/#FLAIR2"
echo "   - Place in: FLAIR-2-main/dataset/"
echo ""
echo "2. Update config with dataset path:"
echo "   - Edit: configs/default.yaml"
echo "   - Set data.root_dir to your dataset location"
echo ""
echo "3. (Optional) Build 10k subset for quick testing:"
echo "   python tools/build_subset.py --root-dir FLAIR-2-main/dataset"
echo ""
echo "4. Start training:"
echo "   python -m src.cli train --config configs/default.yaml"
echo ""
echo "For more details, see IMPLEMENTATION_GUIDE.md"
echo "=========================================="
