import nltk
import os

# Define the download directory relative to this script
download_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')

# Create the directory if it doesn't exist
os.makedirs(download_dir, exist_ok=True)

# Append the directory to NLTK's data path
nltk.data.path.append(download_dir)

# List of packages to download based on pipeline.py
packages = ['punkt', 'stopwords', 'wordnet', 'punkt_tab']

print(f"Downloading NLTK data to: {download_dir}")

for package in packages:
    print(f"Downloading {package}...")
    nltk.download(package, download_dir=download_dir)

print("Download complete.")