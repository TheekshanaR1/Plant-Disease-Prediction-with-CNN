"""
Script to download PlantVillage dataset from Kaggle
"""
import os
import json
from zipfile import ZipFile

print("="*60)
print("PlantVillage Dataset Downloader")
print("="*60)

# Check if kaggle is installed
try:
    import kaggle
    print("\n✓ Kaggle library found")
except ImportError:
    print("\n⚠️ Kaggle library not found. Installing...")
    os.system('pip install kaggle')
    import kaggle
    print("✓ Kaggle library installed")

# Check for Kaggle credentials
kaggle_json_path = os.path.expanduser('~/.kaggle/kaggle.json')
local_kaggle_json = 'kaggle.json'

if not os.path.exists(kaggle_json_path):
    if os.path.exists(local_kaggle_json):
        print(f"\n📋 Found kaggle.json in current directory")
        # Read credentials from local file
        with open(local_kaggle_json, 'r') as f:
            kaggle_credentials = json.load(f)
        
        # Set up environment variables
        os.environ['KAGGLE_USERNAME'] = kaggle_credentials['username']
        os.environ['KAGGLE_KEY'] = kaggle_credentials['key']
        print("✓ Kaggle credentials loaded")
    else:
        print("\n⚠️ Kaggle API credentials not found!")
        print("\nPlease do one of the following:")
        print("1. Create a kaggle.json file in the current directory with your credentials:")
        print('   {"username":"your_username","key":"your_api_key"}')
        print("\n2. OR download the dataset manually from:")
        print("   https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        print("\n3. To get your Kaggle API key:")
        print("   - Go to https://www.kaggle.com/account")
        print("   - Click 'Create New API Token'")
        print("   - Save the downloaded kaggle.json file")
        exit(1)
else:
    print(f"\n✓ Kaggle credentials found at {kaggle_json_path}")

# Download dataset
print("\n📥 Downloading PlantVillage dataset...")
print("This may take several minutes depending on your internet speed...")

try:
    os.system('kaggle datasets download -d abdallahalidev/plantvillage-dataset')
    print("\n✓ Download completed!")
    
    # Unzip the dataset
    zip_file = 'plantvillage-dataset.zip'
    if os.path.exists(zip_file):
        print(f"\n📦 Extracting {zip_file}...")
        with ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall()
        print("✓ Extraction completed!")
        
        # Check the extracted folders
        if os.path.exists('plantvillage dataset'):
            print("\n✓ Dataset extracted successfully!")
            print(f"   Location: plantvillage dataset/")
            
            # Rename to remove space (optional)
            if not os.path.exists('plantvillage_dataset'):
                os.rename('plantvillage dataset', 'plantvillage_dataset')
                print("✓ Renamed to 'plantvillage_dataset'")
            
            # Show dataset info
            base_dir = 'plantvillage_dataset/color'
            if os.path.exists(base_dir):
                num_classes = len(os.listdir(base_dir))
                print(f"\n📊 Dataset Information:")
                print(f"   - Number of disease classes: {num_classes}")
                print(f"   - Dataset path: {base_dir}")
                print(f"\n✅ Dataset is ready for training!")
            else:
                print(f"\n⚠️ Warning: Expected path not found: {base_dir}")
        else:
            print("\n⚠️ Extraction may have failed. Please check manually.")
    else:
        print(f"\n⚠️ Zip file not found: {zip_file}")
        
except Exception as e:
    print(f"\n❌ Error during download: {e}")
    print("\nPlease download the dataset manually from:")
    print("https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
    print("Then extract it to the 'plantvillage_dataset' folder")

print("\n" + "="*60)
