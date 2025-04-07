import torch
from PIL import Image
import numpy as np
from .utils import download_image, image_grid2, imshow

def inferenceFunc(pipe, prompt, base_image, num_inference_steps, image_guidance_scale, guidance_scale, generator):
    edited_image = pipe(prompt, 
        image=base_image, 
        num_inference_steps=num_inference_steps, 
        image_guidance_scale=image_guidance_scale, 
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="np.array",
    ).images[0]
    
    edited_image = mima(edited_image, edited_image.min(), edited_image.max())*255
    edited_image = Image.fromarray(np.uint8(edited_image)).convert('L')
    base_image = mima(base_image, base_image.min(), base_image.max())*255
    base_image = Image.fromarray(np.uint8(base_image).squeeze(0).transpose(1, 2, 0)).convert('L')
    return [base_image, edited_image]

def infer_later_mmse(pipe, fileName, image_guidance_scale, guidance_scale, num_inference_steps):
    base_image = download_image(fileName)
    mmse_values = [13, 15, 20, 27, 29]
    
    for mmse in mmse_values:
        prompt = f"Make it in a tau image with later stage, mmse {mmse}"
        img1 = inferenceFunc(pipe, prompt, base_image, 
                           num_inference_steps, image_guidance_scale, guidance_scale, generator)
        img2 = image_grid2(img1, 1, 2)
        imshow(img2)

def infer_early_mmse(pipe, fileName, image_guidance_scale, guidance_scale, num_inference_steps):
    base_image = download_image(fileName)
    mmse_values = [13, 15, 20, 27, 29]
    
    for mmse in mmse_values:
        prompt = f"Make it in a tau image with early stage, mmse {mmse}"
        img1 = inferenceFunc(pipe, prompt, base_image, 
                           num_inference_steps, image_guidance_scale, guidance_scale, generator)
        img2 = image_grid2(img1, 1, 2)
        imshow(img2)
