#!/usr/bin/env python3
"""
Data Download Script for Torob AI Shopping Assistant
Downloads and extracts competition data from Google Drive
"""

import os
import requests
import tarfile
import logging
from pathlib import Path
import sys
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Google Drive direct download URL for torob-turbo-stage2.tar.gz
GOOGLE_DRIVE_FILE_ID = "1W4mSI33IbeKkWztK3XmE05F7m4tNYDYu"
GOOGLE_DRIVE_URL = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"

# Local paths
DATA_DIR = Path("./data")
ARCHIVE_PATH = Path("./torob-turbo-stage2.tar.gz")
TEMP_EXTRACT_DIR = Path("./temp_extract")

# Expected parquet files
EXPECTED_FILES = [
    "searches.parquet",
    "base_views.parquet", 
    "final_clicks.parquet",
    "base_products.parquet",
    "members.parquet",
    "shops.parquet",
    "categories.parquet",
    "brands.parquet",
    "cities.parquet"
]

def get_confirm_token(response):
    """Extract confirmation token from Google Drive response"""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def download_file_from_google_drive(file_id: str, destination: Path) -> bool:
    """Download large file from Google Drive with virus scan bypass"""
    try:
        session = requests.Session()
        
        # First request to get confirmation token
        logger.info("Getting download confirmation token...")
        response = session.get(GOOGLE_DRIVE_URL, stream=True)
        token = get_confirm_token(response)
        
        # If token exists, make confirmed request
        if token:
            params = {'confirm': token}
            response = session.get(GOOGLE_DRIVE_URL, params=params, stream=True)
        
        # Check if response is successful
        if response.status_code != 200:
            logger.error(f"Failed to download: HTTP {response.status_code}")
            return False
        
        # Download file
        logger.info(f"Downloading data archive to {destination}...")
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024) == 0:  # Log every MB
                            logger.info(f"Download progress: {progress:.1f}% ({downloaded:,}/{total_size:,} bytes)")
        
        logger.info(f"Download completed: {destination} ({downloaded:,} bytes)")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return False

def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """Extract tar.gz archive"""
    try:
        logger.info(f"Extracting {archive_path} to {extract_to}...")
        
        with tarfile.open(archive_path, 'r:gz') as tar:
            # Extract all contents
            tar.extractall(path=extract_to)
            
        logger.info("Archive extracted successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error extracting archive: {e}")
        return False

def move_parquet_files(source_dir: Path, target_dir: Path) -> bool:
    """Find and move parquet files to data directory"""
    try:
        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Find parquet files recursively
        parquet_files = list(source_dir.rglob("*.parquet"))
        
        if not parquet_files:
            logger.error("No parquet files found in extracted archive")
            return False
        
        logger.info(f"Found {len(parquet_files)} parquet files")
        
        # Move files to data directory
        for file_path in parquet_files:
            target_path = target_dir / file_path.name
            logger.info(f"Moving {file_path.name} to data directory")
            
            # Copy file content
            with open(file_path, 'rb') as src, open(target_path, 'wb') as dst:
                dst.write(src.read())
        
        return True
        
    except Exception as e:
        logger.error(f"Error moving parquet files: {e}")
        return False

def verify_data_files(data_dir: Path) -> bool:
    """Verify all expected parquet files exist and are valid"""
    try:
        missing_files = []
        
        for filename in EXPECTED_FILES:
            file_path = data_dir / filename
            if not file_path.exists():
                missing_files.append(filename)
            else:
                # Check file size (should be > 0)
                if file_path.stat().st_size == 0:
                    missing_files.append(f"{filename} (empty)")
        
        if missing_files:
            logger.error(f"Missing or invalid files: {missing_files}")
            return False
        
        logger.info("All data files verified successfully")
        
        # Log file sizes
        for filename in EXPECTED_FILES:
            file_path = data_dir / filename
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"  {filename}: {size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"Error verifying data files: {e}")
        return False

def cleanup_temp_files():
    """Clean up temporary files"""
    try:
        if ARCHIVE_PATH.exists():
            ARCHIVE_PATH.unlink()
            logger.info("Removed archive file")
        
        if TEMP_EXTRACT_DIR.exists():
            import shutil
            shutil.rmtree(TEMP_EXTRACT_DIR)
            logger.info("Removed temporary extraction directory")
            
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")

def main():
    """Main download and setup process"""
    logger.info("Starting Torob data download and setup...")
    
    # Check if data already exists
    if DATA_DIR.exists() and verify_data_files(DATA_DIR):
        logger.info("Data files already exist and are valid. Skipping download.")
        return True
    
    try:
        # Step 1: Download archive
        if not download_file_from_google_drive(GOOGLE_DRIVE_FILE_ID, ARCHIVE_PATH):
            logger.error("Failed to download data archive")
            return False
        
        # Step 2: Extract archive
        if not extract_archive(ARCHIVE_PATH, TEMP_EXTRACT_DIR):
            logger.error("Failed to extract archive")
            return False
        
        # Step 3: Move parquet files
        if not move_parquet_files(TEMP_EXTRACT_DIR, DATA_DIR):
            logger.error("Failed to move parquet files")
            return False
        
        # Step 4: Verify files
        if not verify_data_files(DATA_DIR):
            logger.error("Data verification failed")
            return False
        
        # Step 5: Cleanup
        cleanup_temp_files()
        
        logger.info("Data setup completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Unexpected error during setup: {e}")
        return False
    
    finally:
        # Always attempt cleanup
        cleanup_temp_files()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)