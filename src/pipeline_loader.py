import torch
from diffusers import StableDiffusionAdapterPipeline, T2IAdapter, AutoencoderKL

def load_semantic_pipeline():
    print("🚀 Loading Models in FULL PRECISION (Stability Mode)...")
    
    # Load everything in float32
    adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2iadapter_seg_sd14v1", 
        torch_dtype=torch.float32
    )

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", 
        torch_dtype=torch.float32
    )

    pipe = StableDiffusionAdapterPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V6.0_B1_noVAE",
        adapter=adapter,
        vae=vae,
        torch_dtype=torch.float32  # Nuclear option: no float16
    )

    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    pipe.to("mps")
    # Slicing is extra important in float32 to save RAM
    pipe.enable_attention_slicing() 
    
    print("✅ Full Precision Pipeline Ready!")
    return pipe