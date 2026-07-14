import os
import sys
import torch
from tqdm import tqdm
from diffusers.utils import load_image

# 1. Path Setup
current_script = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pipeline_loader import load_semantic_pipeline

def main():
    # 2. Setup
    pipe = load_semantic_pipeline()
    
    # Define source folders
    png_folder = os.path.join(project_root, "data", "dataset", "PNG")
    raw_mixed_folder = os.path.join(project_root, "data", "raw_mixed")
    
    output_dir = os.path.join(project_root, "outputs", "batch_results")
    os.makedirs(output_dir, exist_ok=True)

    # 3. Get all unique masks from EVERY folder inside 'data'
    all_masks_dict = {}
    
    # Extensions to look for
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')

    # This walks through every single subfolder in 'data' to find your 1,132 images
    for root, dirs, files in os.walk(os.path.join(project_root, "data")):
        # Skip the outputs folder so we don't process our own results
        if "outputs" in root:
            continue
            
        for f in files:
            if f.lower().endswith(valid_extensions):
                # Use filename as key, absolute path as value
                if f not in all_masks_dict:
                    all_masks_dict[f] = os.path.join(root, f)

    sorted_mask_names = sorted(all_masks_dict.keys())
    
    print("-" * 30)
    print(f"📂 Total unique masks found across all folders: {len(sorted_mask_names)}")
    if len(sorted_mask_names) < 1132:
        print(f"⚠️ Warning: Still only found {len(sorted_mask_names)} out of 1,132.")
    print("-" * 30)

    # 4. Parameters
    prompt = (
        "A high-end modern architectural building facade, "
        "professional architectural photography, 8k resolution, "
        "photorealistic, cinematic sunset lighting, sharp focus, highly detailed"
    )
    negative_prompt = "blurry, low quality, distorted, cartoon, anime, drawing, painting, deformed, messy"

    # 5. The Loop
    for mask_name in tqdm(sorted_mask_names, desc="Processing Remaining"):
        save_path = os.path.join(output_dir, f"gen_{mask_name}")
        
        # SKIP if already done (Resume capability)
        if os.path.exists(save_path):
            continue

        # Get the correct path for this specific mask
        mask_path = all_masks_dict[mask_name]
        
        try:
            # Prepare image
            control_image = load_image(mask_path).resize((512, 512))

            with torch.inference_mode():
                # Using the Full Precision (float32) pipeline we set up earlier
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=control_image,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    adapter_conditioning_scale=1.0 
                ).images[0]
                
                image.save(save_path)
                
        except Exception as e:
            print(f"\n❌ Error on {mask_name}: {e}")
            continue

    print(f"✅ Finished! All results are in: {output_dir}")

if __name__ == "__main__":
    main()