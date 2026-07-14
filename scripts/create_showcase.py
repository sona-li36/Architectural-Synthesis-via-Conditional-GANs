import os
from PIL import Image

def create_grid(mask_name):
    project_root = os.getcwd()
    mask_path = os.path.join(project_root, "data", "dataset", "PNG", mask_name)
    gen_path = os.path.join(project_root, "outputs", "batch_results", f"gen_{mask_name}")
    save_dir = os.path.join(project_root, "outputs", "showcase")
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(mask_path) and os.path.exists(gen_path):
        m = Image.open(mask_path).convert("RGB").resize((512, 512))
        g = Image.open(gen_path).convert("RGB").resize((512, 512))
        
        canvas = Image.new("RGB", (1024, 512))
        canvas.paste(m, (0, 0))
        canvas.paste(g, (512, 0))
        canvas.save(os.path.join(save_dir, f"compare_{mask_name}"))
        print(f"✅ Created grid for {mask_name}")

# Pick 10 specific masks or just take the first 10
masks_to_show = ["cmp_b0001.png", "cmp_b0002.png", "cmp_b0003.png"] # Add your best ones here
for m in masks_to_show:
    create_grid(m)