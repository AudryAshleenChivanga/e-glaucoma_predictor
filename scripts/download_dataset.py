"""
Dataset Download Script for Glaucoma E-Predictor
Downloads the glaucoma detection dataset from Kaggle and prepares it for training.
"""
import os
import shutil
import sys
from pathlib import Path


def download_dataset():
    """Download the glaucoma detection dataset from Kaggle."""
    try:
        import kagglehub
    except ImportError:
        print("Installing kagglehub...")
        os.system(f"{sys.executable} -m pip install kagglehub")
        import kagglehub
    
    print("=" * 60)
    print("Glaucoma Detection Dataset Downloader")
    print("=" * 60)
    
    # Download the dataset
    print("\n📥 Downloading dataset from Kaggle...")
    print("   Dataset: sshikamaru/glaucoma-detection")
    
    try:
        path = kagglehub.dataset_download("sshikamaru/glaucoma-detection")
        print(f"\n✅ Dataset downloaded to: {path}")
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\n💡 Make sure you have Kaggle credentials configured:")
        print("   1. Create a Kaggle account at https://www.kaggle.com")
        print("   2. Go to Account -> Create New API Token")
        print("   3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<User>\\.kaggle\\ (Windows)")
        return None
    
    return path


def setup_data_directory(source_path: str, target_dir: str = "data"):
    """
    Copy and organize the dataset into the expected directory structure.
    
    Expected structure:
    data/
    ├── Glaucoma/       (or similar name for positive cases)
    │   ├── image1.jpg
    │   └── ...
    └── Normal/         (or similar name for negative cases)
        ├── image1.jpg
        └── ...
    """
    print(f"\n📁 Setting up data directory at: {target_dir}")
    
    source = Path(source_path)
    target = Path(target_dir)
    
    # Create target directory
    target.mkdir(parents=True, exist_ok=True)
    
    # Find and copy data
    copied_count = 0
    
    # Check for common directory structures
    for item in source.rglob("*"):
        if item.is_dir():
            dir_name = item.name.lower()
            
            # Look for glaucoma/positive cases
            if any(key in dir_name for key in ['glaucoma', 'positive', 'disease', 'abnormal']):
                target_subdir = target / "glaucoma"
                target_subdir.mkdir(exist_ok=True)
                
                for img in item.glob("*"):
                    if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                        shutil.copy2(img, target_subdir / img.name)
                        copied_count += 1
            
            # Look for normal/negative cases
            elif any(key in dir_name for key in ['normal', 'negative', 'healthy', 'non']):
                target_subdir = target / "normal"
                target_subdir.mkdir(exist_ok=True)
                
                for img in item.glob("*"):
                    if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                        shutil.copy2(img, target_subdir / img.name)
                        copied_count += 1
    
    # If no specific structure found, try to copy from root
    if copied_count == 0:
        print("   Looking for alternative directory structure...")
        
        # Try direct subdirectories
        for subdir in source.iterdir():
            if subdir.is_dir():
                target_subdir = target / subdir.name
                target_subdir.mkdir(exist_ok=True)
                
                for img in subdir.glob("*"):
                    if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                        shutil.copy2(img, target_subdir / img.name)
                        copied_count += 1
    
    print(f"\n✅ Copied {copied_count} images to {target}")
    
    # Display directory structure
    print("\n📂 Data directory structure:")
    for subdir in sorted(target.iterdir()):
        if subdir.is_dir():
            img_count = len(list(subdir.glob("*")))
            print(f"   {subdir.name}/: {img_count} images")
    
    return target


def main():
    """Main function to download and setup the dataset."""
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Download dataset
    source_path = download_dataset()
    
    if source_path:
        # Setup data directory
        target_dir = project_root / "data"
        setup_data_directory(source_path, str(target_dir))
        
        print("\n" + "=" * 60)
        print("🎉 Dataset setup complete!")
        print("=" * 60)
        print(f"\nData location: {target_dir}")
        print("\nNext steps:")
        print("1. Start the backend: cd backend && python main.py")
        print("2. Open the frontend in a browser: frontend/index.html")
        print("3. Train the model using the Admin panel or API")
    else:
        print("\n⚠️  Dataset download failed. Please try again.")


if __name__ == "__main__":
    main()

