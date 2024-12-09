import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import shutil

def preprocess_image(image):
    """Preprocess single image"""
    if image is None:
        return None
    
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to standard size
    image = cv2.resize(image, (224, 224))
    
    # Normalize to [0,1]
    image = image.astype(np.float32) / 255.0
    
    return image

def process_dataset(raw_dir, processed_dir):
    """Process dataset maintaining directory structure"""
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    
    # Ensure processed directory exists
    processed_path.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ['train', 'validation', 'test']:
        print(f"Processing {split} split...")
        
        # Create output directories
        for label in ['RG', 'NRG']:
            out_dir = processed_path / split / label
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Process all images in current split/label directory
            in_dir = raw_path / split / label
            if not in_dir.exists():
                print(f"Warning: Directory {in_dir} does not exist")
                continue
                
            for img_path in tqdm(list(in_dir.glob('*.jpg'))):
                try:
                    # Read and preprocess image
                    image = cv2.imread(str(img_path))
                    if image is None:
                        print(f"Warning: Could not read {img_path}")
                        continue
                        
                    processed = preprocess_image(image)
                    if processed is None:
                        continue
                    
                    # Save processed image
                    out_path = out_dir / img_path.name
                    cv2.imwrite(str(out_path), (processed * 255).astype(np.uint8))
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    
    # Process dataset
    process_dataset(raw_dir, processed_dir)
    print("Processing completed!")