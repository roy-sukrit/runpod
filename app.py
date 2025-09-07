from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch, io, base64
from PIL import Image

class GenerateRequest(BaseModel):
    prompt: str

app = FastAPI(title="LoRA Image Generation API")

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    "dreamlike-art/dreamlike-photoreal-2.0",
    dtype=torch.float16 if device=="cuda" else torch.float32
).to(device)

pipe.load_lora_weights("./", weight_name="pytorch_lora_weights_66_img.safetensors", prefix=None)
pipe.fuse_lora(lora_scale=0.8)

@app.post("/generate")
def generate_image(req: GenerateRequest):
    with torch.no_grad():
        image = pipe(req.prompt).images[0]
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return {"image_base64": img_str}