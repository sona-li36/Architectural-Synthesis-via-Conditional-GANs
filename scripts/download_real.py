import os
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import time

def download_architectural_photos(count=50): # Let's try 50 first to be safe
    project_root = os.getcwd()
    target_dir = os.path.join(project_root, "data", "real_reference")
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"📥 Downloading {count} real architectural photos...")

    # A more reliable direct link to an architectural photo source
    # We use a User-Agent to pretend we are a browser like Chrome
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    downloaded = 0
    for i in tqdm(range(count)):
        try:
            # Using a reliable placeholder service for architecture
            img_url = f"https://loremflickr.com/512/512/architecture,building?lock={i}"
            response = requests.get(img_url, headers=headers, timeout=20)
            
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert('RGB')
                save_path = os.path.join(target_dir, f"real_{i:04d}.png")
                img.save(save_path)
                downloaded += 1
            
            time.sleep(1) # Wait 1 second so we don't get banned
            
        except Exception as e:
            print(f"Error on image {i}: {e}")
            continue

    print(f"\n✅ Done! Successfully saved {downloaded} images to {target_dir}")

if __name__ == "__main__":
    download_architectural_photos(50)