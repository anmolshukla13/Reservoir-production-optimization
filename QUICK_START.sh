#!/bin/bash
# Quick Start Script for Reservoir Production Optimization

echo "🛢️ Reservoir Production Optimization - Quick Start"
echo "=================================================="
echo ""

# Check Python version
python3 --version || { echo "❌ Python 3.9+ required"; exit 1; }

echo "✓ Python found"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Generate data:     python data_generator.py"
echo "2. Train models:      python model_training.py"
echo "3. Start API:         uvicorn api_main:app --reload"
echo "4. Start dashboard:   streamlit run dashboard.py"
echo ""
echo "🚀 Happy optimizing!"
