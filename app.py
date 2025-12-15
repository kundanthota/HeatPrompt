# app.py
from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import pyproj
import os
from datetime import datetime

import os, io, json, base64, urllib.parse, datetime, uuid
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError, ImageDraw

import yaml
import joblib
from pydantic import BaseModel
from typing import List, Optional

from dotenv import load_dotenv
import requests

# ────────────────────────────────────────────────────────────────
# Setup
# ────────────────────────────────────────────────────────────────
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Config (prompt)
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)
system_prompt = config["system"]["prompt"]
temperature = float(config["system"].get("temperature", 0.1))

# OpenRouter
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
OR_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
# Optional model
model = None
model_path = "models/best_gpt4o_regressor.joblib"
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        print(f"[INFO] Loaded model: {model_path}")
    except Exception as e:
        print(f"[WARN] Failed to load model: {e}")

# Optional embedding (safe import)
sentence_model, torch_F = None, None
try:
    from sentence_transformers import SentenceTransformer
    import torch.nn.functional as F
    sentence_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    torch_F = F
    print("[INFO] Loaded sentence-transformer.")
except Exception as e:
    print(f"[WARN] Embedding model not available: {e}")


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_caption_from_image(image_path: str) -> str:
    data_url = f"data:image/png;base64,{encode_image_to_base64(image_path)}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]
    payload = {"model": "openai/gpt-4o", "temperature": temperature, "messages": messages}
    r = requests.post(OR_URL, headers=OR_HEADERS, json=payload, timeout=60)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"]

def reproject_3857_to_4326(coords_3857: list) -> list:
    """
    Convert [x, y] Web Mercator to [lon, lat] WGS84
    """
    project = pyproj.Transformer.from_crs(3857, 4326, always_xy=True).transform
    projected = [project(x, y) for x, y in coords_3857]
    return projected

def polygon_to_overpass_coords(coords: list) -> str:
    return " ".join(f"{lat} {lon}" for lon, lat in coords)

def query_overpass_building_count(coords_3857: list) -> int:
    # Convert to EPSG:4326
    coords_4326 = reproject_3857_to_4326(coords_3857)
    overpass_poly = polygon_to_overpass_coords(coords_4326)

    query = f"""
    [out:json][timeout:25];
    (
      way["building"](poly:"{overpass_poly}");
    );
    out body;
    """
    response = requests.post(OVERPASS_URL, data={"data": query})
    response.raise_for_status()
    data = response.json()
    return len(data.get("elements", []))

# ────────────────────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": str(datetime.datetime.now())}



@app.post("/upload-image")
async def process_image(image: UploadFile = File(...),
    coordinates: str = Form(...),
    length_meters: float = Form(...),
    area_sq_meters: float = Form(...)):
    try:

        with open('image.png', "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        # Caption
        caption = get_caption_from_image('image.png')
        
        sent_emb = sentence_model.encode([caption], convert_to_tensor=True)
        sent_emb = F.layer_norm(sent_emb, normalized_shape=(sent_emb.shape[1],))
        sent_emb = sent_emb[:, :512]
        embedding= F.normalize(sent_emb, p=2, dim=1).cpu().numpy()[0].tolist()

        coords = json.loads(coordinates)
        ring = coords[0]  # Assuming single ring polygon

        try:
            building_count = query_overpass_building_count(ring)
        except Exception as e:
            return JSONResponse(status_code=500, content={
                "status": "error",
                "message": f"Overpass API failed: {str(e)}"
            })
        prediction = None
        if model is not None and embedding is not None:
            # include optional meta features if provided, else 0

            features = embedding + [length_meters, area_sq_meters, building_count]
            try:
                prediction = float(model.predict([features])[0])/10
            except Exception as e:
                print(f"[WARN] Prediction failed: {e}")
                prediction = None
        return {
            "caption": caption,
            "prediction": prediction
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[ERROR] process_image: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"{e} — see server logs")


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)