# main.py
import torch
import matplotlib.pyplot as plt
from CustomInstructPix2Pix import StableDiffusionInstructPix2PixPipeline_SI
from src.inference import infer_later_mmse, infer_early_mmse

def main():
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model setup
    model_id = "sijang/taupetgen"
    
    # Initialize model
    pipe = StableDiffusionInstructPix2PixPipeline_SI.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    
    generator = torch.Generator(device).manual_seed(0)
    pipe.safety_checker = lambda images, clip_input: (images, False)

    # Parameters
    image_guidance_scale = 1.5
    guidance_scale = 2
    num_inference_steps = 10
    fileName = "example/mr_example1.png"

    # Generate images
    infer_later_mmse(pipe, fileName, image_guidance_scale, guidance_scale, num_inference_steps, generator)
    
    # Display all generated images
    plt.show()

if __name__ == "__main__":
    main()
