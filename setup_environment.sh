#!/bin/bash

# EEMD-TGNet Environment Setup Script
# This script sets up the Python environment for running the EEMD-TGNet experiments

echo "🚀 Setting up EEMD-TGNet Environment..."
echo "=================================================="

# Check if conda is installed
if command -v conda &> /dev/null
then
    echo "✅ Conda found. Setting up conda environment..."
    
    # Create conda environment
    conda env create -f environment.yml
    
    echo "✅ Conda environment 'eemd-tgnet' created successfully!"
    echo ""
    echo "To activate the environment, run:"
    echo "conda activate eemd-tgnet"
    echo ""
    echo "Then start Jupyter:"
    echo "jupyter notebook"
    
else
    echo "❌ Conda not found. Using pip instead..."
    
    # Check if pip is installed
    if command -v pip &> /dev/null
    then
        echo "✅ Pip found. Installing requirements..."
        
        # Create virtual environment (optional but recommended)
        python -m venv eemd-tgnet-env
        
        # Activate virtual environment
        source eemd-tgnet-env/bin/activate
        
        # Install requirements
        pip install -r requirements.txt
        
        echo "✅ Virtual environment created and packages installed!"
        echo ""
        echo "To activate the environment, run:"
        echo "source eemd-tgnet-env/bin/activate"
        echo ""
        echo "Then start Jupyter:"
        echo "jupyter notebook"
        
    else
        echo "❌ Neither conda nor pip found. Please install Python package manager."
        exit 1
    fi
fi

echo ""
echo "🎉 Environment setup complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Activate the environment (see instructions above)"
echo "2. Start Jupyter notebook"
echo "3. Run notebooks in order:"
echo "   - 01_Data_Generation_EEMD_Decomposition.ipynb"
echo "   - 02_TCN_GRU_Architecture_Training.ipynb"
echo "   - 03_Evaluation_Results_Analysis.ipynb"
echo ""
echo "📖 See README_EEMD_TGNet_Experiments.md for detailed instructions"
