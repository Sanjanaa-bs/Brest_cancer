# File: setup_project.py

import os

def create_project_structure():
    """Create the complete project directory structure"""
    
    directories = [
        'data/raw/GEO',
        'data/raw/TCGA',
        'data/processed',
        'data/markers',
        'src',
        'models',
        'results/figures',
        'results/tables',
        'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    print("\n✅ Project structure created successfully!")

if __name__ == "__main__":
    create_project_structure()