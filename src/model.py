import torch
from CustomInstructPix2Pix import StableDiffusionInstructPix2PixPipeline_SI

def setup_model(model_id="sijang/taupetgen", device="cuda:0"):
    pipe = StableDiffusionInstructPix2PixPipeline_SI.from_pretrained(
        model_id, 
        torch_dtype=torch.float16
    ).to(device)
    
    generator = torch.Generator(device).manual_seed(0)
    pipe.safety_checker = lambda images, clip_input: (images, False)
    
    return pipe, generator
