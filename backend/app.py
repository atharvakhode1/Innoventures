import io
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

app = FastAPI(title="Segformer Inference API")

# Configure CORS so the React app can talk to it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= LOAD MODEL AT STARTUP =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading model on:", device)

# Load real_model_v2.pth from the frontend root
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "real_model_v2.pth")

try:
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=11,
        ignore_mismatched_sizes=True,
        use_safetensors=True
    ).to(device)

    # Load custom weights with strict=False in case of dimension mismatches
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    processor = SegformerImageProcessor()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    processor = None

# ================= COLOR MAP =================
COLOR_MAP = np.array([
    [0, 0, 0],          # Background
    [34, 139, 34],     # Trees
    [0, 200, 0],       # Lush Bushes
    [210, 180, 140],   # Dry Grass
    [139, 90, 43],     # Dry Bushes
    [128, 128, 0],     # Ground Clutter
    [255, 215, 0],     # Flowers
    [139, 69, 19],     # Logs
    [128, 128, 128],   # Rocks
    [160, 82, 45],     # Landscape
    [135, 206, 235],   # Sky
], dtype=np.uint8)

@app.post("/predict")
async def predict_image(image: UploadFile = File(...)):
    if not model:
        return Response(content="Model not loaded server-side", status_code=500)
    
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        inputs = processor(images=img, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = model(**inputs)
            
        pred = F.interpolate(
            out.logits,
            size=img.size[::-1],
            mode="bilinear",
            align_corners=False
        )
        
        pred = pred.argmax(1).squeeze().cpu().numpy()
        color_pred = COLOR_MAP[pred]
        
        result_img = Image.fromarray(color_pred)
        img_byte_arr = io.BytesIO()
        result_img.save(img_byte_arr, format='PNG')
        
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")
    except Exception as e:
        print("Prediction error:", e)
        return Response(content=str(e), status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
