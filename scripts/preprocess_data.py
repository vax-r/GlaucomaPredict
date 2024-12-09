import os
from pathlib import Path
from src.data.preprocessing import process_dataset

def main():
    # Get project root directory
    project_root = Path(__file__).parent.parent
    
    # Define data directories
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    
    print(f"Processing images from {raw_dir}")
    print(f"Saving results to {processed_dir}")
    
    # Run preprocessing
    process_dataset(raw_dir, processed_dir)

if __name__ == "__main__":
    main()