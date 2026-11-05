import os
import io
import asyncio
import requests
from PIL import Image, ImageDraw

def _run_hf_router_sync(image_path: str) -> bytes:
    # Read your exact environment variable name
    hf_token = os.environ.get("HF_API_TOKEN")
    
    if not hf_token:
        raise ValueError("HF_API_TOKEN environment variable is missing.")

    # 1. Load original image and calculate expansion (20% padding)
    original = Image.open(image_path).convert("RGB")
    orig_w, orig_h = original.size
    
    pad_x = int(orig_w * 0.20)
    pad_y = int(orig_h * 0.20)
    new_w = orig_w + (pad_x * 2)
    new_h = orig_h + (pad_y * 2)
    
    # 2. Create the expanded canvas (white borders)
    expanded_img = Image.new("RGB", (new_w, new_h), (255, 255, 255))
    expanded_img.paste(original, (pad_x, pad_y))
    
    # 3. Create the mask (Black = preserve original, White = inpaint borders)
    mask = Image.new("L", (new_w, new_h), 255)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([pad_x, pad_y, pad_x + orig_w, pad_y + orig_h], fill=0)
    
    # 4. Convert images to bytes for the payload
    img_bytes = io.BytesIO()
    expanded_img.save(img_bytes, format="PNG")
    
    mask_bytes = io.BytesIO()
    mask.save(mask_bytes, format="PNG")
    
    # 5. Target the modern Hugging Face Inference Router endpoint directly
    api_url = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-2-inpainting"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    files = {
        'image': ('image.png', img_bytes.getvalue(), 'image/png'),
        'mask_image': ('mask.png', mask_bytes.getvalue(), 'image/png')
    }
    data = {
        'inputs': 'Seamless photorealistic front yard, green lawn, concrete driveway, clear blue sky, neighborhood street'
    }
    
    print(f"[Cloud AI] Sending {image_path} to modern HF router for outpainting...", flush=True)
    response = requests.post(api_url, headers=headers, files=files, data=data)
    
    if response.status_code != 200:
        raise RuntimeError(f"HF Router Error ({response.status_code}): {response.text}")
        
    return response.content

async def expand_image_borders(image_path: str) -> str:
    """
    Calls the modern Hugging Face Inference Router via direct HTTP to generate 
    expanded site context imagery for DUSt3R.
    """
    output_bytes = await asyncio.to_thread(_run_hf_router_sync, image_path)
    
    outpainted_path = image_path.replace(".png", "_expanded.png")
    with open(outpainted_path, "wb") as f:
        f.write(output_bytes)
        
    print(f"[Cloud AI] ✅ Outpainting complete: {outpainted_path}", flush=True)
    return outpainted_path