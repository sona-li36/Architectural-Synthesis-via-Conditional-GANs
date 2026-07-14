import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def calculate_fidelity(mask_path, gen_path):
    # Load and normalize
    mask = np.array(Image.open(mask_path).convert("RGB").resize((128, 128)))
    gen = np.array(Image.open(gen_path).convert("RGB").resize((128, 128)))
    
    # Calculate Pixel-wise correlation
    # This measures how well the AI placed colors in the right spots
    correlation = np.corrcoef(mask.flatten(), gen.flatten())[0, 1]
    return max(0, correlation * 100)

def main():
    project_root = os.getcwd()
    mask_folder = os.path.join(project_root, "data", "dataset", "PNG")
    gen_folder = os.path.join(project_root, "outputs", "batch_results")
    
    # Get all generated files
    generated_files = [f for f in os.listdir(gen_folder) if f.startswith("gen_")]
    
    scores = []
    print(f"📊 Analyzing {len(generated_files)} images for Structural Fidelity...")

    for gen_name in tqdm(generated_files):
        # Extract original mask name (removes 'gen_')
        mask_name = gen_name.replace("gen_", "")
        mask_path = os.path.join(mask_folder, mask_name)
        gen_path = os.path.join(gen_folder, gen_name)
        
        if os.path.exists(mask_path):
            score = calculate_fidelity(mask_path, gen_path)
            scores.append(score)

    if scores:
        avg_score = sum(scores) / len(scores)
        print("\n" + "="*30)
        print(f"✅ EVALUATION COMPLETE")
        print(f"📈 Average Structural Fidelity: {avg_score:.2f}%")
        print(f"⭐ Best Match: {max(scores):.2f}%")
        print("="*30)

if __name__ == "__main__":
    main()