## Prototype Development for Image Generation Using the Stable Diffusion Model and Gradio Framework

### AIM:
To design and deploy a prototype application for image generation utilizing the Stable Diffusion model, integrated with the Gradio UI framework for interactive user engagement and evaluation.

### PROBLEM STATEMENT:
Generating realistic images from text prompts is a complex task that typically requires advanced machine learning knowledge. This project aims to simplify the process by creating an interactive prototype that allows users to generate images using the Stable Diffusion model. By integrating the model with a Gradio interface, users can enter a text prompt and instantly receive a high-quality image, making AI-based image generation accessible and easy to use.

### DESIGN STEPS:
#### STEP 1:
Use the pre-trained Stable Diffusion model available from Hugging Face's diffusers library. Ensure all dependencies, including GPU acceleration (if available), are set up for optimal performance.

#### STEP 2:
Use Gradio to create an interface with: Input: Textbox for entering the prompt. Output: Display panel for the generated image.

#### STEP 3:
Load the Stable Diffusion model and tokenizer. Accept a text prompt from the user via Gradio's input. Preprocess the text and generate an image using Stable Diffusion. Display the generated image in the output panel.

#### STEP 4:
Test the application with diverse prompts to ensure quality. Deploy the application on a public platform (e.g., Hugging Face Spaces or local Gradio server).

### PROGRAM:
```
import os
import io
import IPython.display
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']


import gradio as gr 

#A helper function to convert the PIL image to base64
#so you can send it to the API
def base64_to_pil(img_base64):
    base64_decoded = base64.b64decode(img_base64)
    byte_stream = io.BytesIO(base64_decoded)
    pil_image = Image.open(byte_stream)
    return pil_image

def generate(prompt):
    output = get_completion(prompt)
    result_image = base64_to_pil(output)
    return result_image

gr.close_all()
demo = gr.Interface(fn=generate,
                    inputs=[gr.Textbox(label="Your prompt")],
                    outputs=[gr.Image(label="Result")],
                    title="Image Generation with Stable Diffusion",
                    description="Generate any image with Stable Diffusion",
                    allow_flagging="never",
                    examples=["the spirit of a tamagotchi wandering in the city of Vienna","a mecha robot in a favela"])

demo.launch(share=True, server_port=int(os.environ['PORT1']))
```

```
import gradio as gr 

#A helper function to convert the PIL image to base64 
# so you can send it to the API
def base64_to_pil(img_base64):
    base64_decoded = base64.b64decode(img_base64)
    byte_stream = io.BytesIO(base64_decoded)
    pil_image = Image.open(byte_stream)
    return pil_image

def generate(prompt, negative_prompt, steps, guidance, width, height):
    params = {
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "width": width,
        "height": height
    }
    
    output = get_completion(prompt, params)
    pil_image = base64_to_pil(output)
    return pil_image


gr.close_all()
demo = gr.Interface(fn=generate,
                    inputs=[
                        gr.Textbox(label="Your prompt"),
                        gr.Textbox(label="Negative prompt"),
                        gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=25,
                                 info="In how many steps will the denoiser denoise the image?"),
                        gr.Slider(label="Guidance Scale", minimum=1, maximum=20, value=7, 
                                  info="Controls how much the text prompt influences the result"),
                        gr.Slider(label="Width", minimum=64, maximum=512, step=64, value=512),
                        gr.Slider(label="Height", minimum=64, maximum=512, step=64, value=512),
                    ],
                    outputs=[gr.Image(label="Result")],
                    title="Image Generation with Stable Diffusion",
                    description="Generate any image with Stable Diffusion",
                    allow_flagging="never"
                    )

demo.launch(share=True, server_port=int(os.environ['PORT2']))
```
### OUTPUT:


<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/ecd9b9d8-3ccb-4eb6-90d0-07e4bce5c648" />


<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/fbef0d1c-b491-4f96-bec8-58d123d0f22a" />




### RESULT:

The application successfully generates high-quality images based on user-provided text prompts. The Stable Diffusion model ensures visually appealing results, and the Gradio interface makes it accessible and interactive.
