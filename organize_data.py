import os
import shutil

# Based on your screenshot, the files are directly in 'data'
SOURCE_DIR = "data" 
TARGET_DIR = "data/dataset"

def organize_dataset():
    extensions = ['jpg', 'png', 'bmp']
    
    # 1. Create the target folders if they don't exist
    for ext in extensions:
        path = os.path.join(TARGET_DIR, ext.upper())
        os.makedirs(path, exist_ok=True)
        print(f"📁 Ensured folder exists: {path}")

    # 2. Iterate through the 'data' folder
    count = 0
    for filename in os.listdir(SOURCE_DIR):
        # We only want to move files, not the subfolders themselves
        if os.path.isdir(os.path.join(SOURCE_DIR, filename)):
            continue
            
        file_ext = filename.split('.')[-1].lower()
        
        if file_ext in extensions:
            src_path = os.path.join(SOURCE_DIR, filename)
            dst_path = os.path.join(TARGET_DIR, file_ext.upper(), filename)
            
            try:
                shutil.move(src_path, dst_path)
                count += 1
            except Exception as e:
                print(f"❌ Error moving {filename}: {e}")

    print(f"\n✅ Done! Successfully organized {count} files into '{TARGET_DIR}'")

if __name__ == "__main__":
    organize_dataset()