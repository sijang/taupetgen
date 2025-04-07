import torch
from CustomInstructPix2Pix import StableDiffusionInstructPix2PixPipeline_SI
from src.inference import infer_later_mmse, infer_early_mmse

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_id = "sijang/taupetgen"
    
    # Initialize model
    pipe = StableDiffusionInstructPix2PixPipeline_SI.from_pretrained(
        model_id, 
        torch_dtype=torch.float16
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
    # Uncomment below to run early MMSE inference
    # infer_early_mmse(pipe, fileName, image_guidance_scale, guidance_scale, num_inference_steps, generator)

if __name__ == "__main__":
    main()
