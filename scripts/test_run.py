import os
import sys
import torch
from diffusers.utils import load_image

# ==========================================
# 1. ATOMIC PATH OVERRIDE
# ==========================================
current_script = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script)
project_root = os.path.dirname(scripts_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.pipeline_loader import load_semantic_pipeline
    print("✅ System Path linked successfully.")
except ImportError as e:
    print("\n❌ STILL UNABLE TO FIND SRC FOLDER ❌")
    sys.exit(1)

def main():
    # 2. Hardware Check
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🖥️  Using Hardware: {device.upper()}")

    # 3. Load Pipeline
    try:
        pipe = load_semantic_pipeline()
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return

    # 4. Locate Data
    png_folder = os.path.join(project_root, "data", "dataset", "PNG")
    all_masks = sorted([f for f in os.listdir(png_folder) if f.lower().endswith('.png')])
    
    if not all_masks:
        print(f"❌ Error: No .png files found in {png_folder}")
        return
        
    test_mask = all_masks[0] 
    mask_path = os.path.join(png_folder, test_mask)
    print(f"🖼️  Processing Layout: {test_mask}")

    # 5. Prepare Image
    control_image = load_image(mask_path).resize((512, 512))

    # 6. Set Visual Parameters
    prompt = (
        "A high-end modern architectural building facade, "
        "professional architectural photography, 8k resolution, "
        "photorealistic, cinematic sunset lighting, sharp focus, highly detailed"
    )
    negative_prompt = (
        "blurry, low quality, distorted, cartoon, anime, drawing, painting, deformed, messy"
    )

    # 7. Execute Generation (Full Precision Mode)
    print("🎨 Generating... (Running in float32 for 100% stability)")
    
    try:
        with torch.inference_mode():
            # In full precision, we use the standard pipeline call
            # This is slower but prevents mathematical "Black Holes"
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control_image,
                num_inference_steps=25,
                guidance_scale=7.5,
                adapter_conditioning_scale=1.0 
            ).images[0]

        # 8. Save result
        output_dir = os.path.join(project_root, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        # We'll use a new prefix so you can compare files
        save_path = os.path.join(output_dir, f"full_precision_{test_mask}")
        output.save(save_path)
        
        print("-" * 30)
        print(f"✅ SUCCESS! Architectural image saved.")
        print(f"📍 Location: {save_path}")
        print("-" * 30)

    except Exception as e:
        print(f"❌ Generation failed: {e}")

if __name__ == "__main__":
    main()