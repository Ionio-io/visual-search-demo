import os
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, AutoModel
from torch.nn.functional import cosine_similarity
from typing import List, Dict, Any
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import time

app = FastAPI()

# Set up CORS to allow frontend to communicate with our API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor
model = AutoModel.from_pretrained(
    "google/siglip-so400m-patch14-384",
    torch_dtype=torch.float16,
    attn_implementation="sdpa",
    device_map=device
).eval().to(device)

processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

# Global cache for image embeddings
embedding_cache: Dict[str, Dict[str, Any]] = {}

def get_image_embeddings(image: Image.Image, model: AutoModel, processor: AutoProcessor) -> torch.Tensor:
    """Helper function to compute image embeddings."""
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        return model.get_image_features(**inputs)

def load_default_images() -> List[str]:
    """Load paths of default images from the directory."""
    target_paths = []
    default_dir = "./default_images"
    if os.path.exists(default_dir):
        for file in os.listdir(default_dir):
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                target_paths.append(os.path.join(default_dir, file))
    return target_paths

class SearchResult(BaseModel):
    path: str
    similarity: float

@app.post("/query_image")
async def query_image(file: UploadFile = File(...)):
    # Process query image
    query_img = Image.open(file.file).convert("RGB")
    query_emb = get_image_embeddings(query_img, model, processor)
    
    # Get target image paths and update cache
    target_paths = load_default_images()
    if not target_paths:
        return JSONResponse(content={"error": "No images found in default database!"}, status_code=404)
    
    # Prune cache for deleted images
    global embedding_cache
    current_paths = set(target_paths)
    for path in list(embedding_cache.keys()):
        if path not in current_paths:
            del embedding_cache[path]
    
    # Process target images with caching
    similarities = []
    for path in target_paths:
        try:
            # Get current modification time
            current_mtime = os.path.getmtime(path)
            
            # Check cache validity
            cached = embedding_cache.get(path)
            if cached and cached["mtime"] == current_mtime:
                emb = cached["embedding"]
            else:
                # Compute and cache embedding if not valid
                img = Image.open(path).convert("RGB")
                emb = get_image_embeddings(img, model, processor)
                embedding_cache[path] = {
                    "embedding": emb,
                    "mtime": current_mtime
                }
            
            # Calculate similarity
            sim = cosine_similarity(query_emb, emb).item()
            similarities.append((sim, path))
            
        except Exception as e:
            return JSONResponse(content={"error": f"Error processing {path}: {str(e)}"}, status_code=500)
    
    # Return top 3 results
    top_3_matches = sorted(similarities, key=lambda x: x[0], reverse=True)[:3]
    results = [SearchResult(path=path, similarity=float(score)) for score, path in top_3_matches]
    
    return {"results": results}

@app.post("/add_image")
async def add_image(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    # Save to default images directory
    default_dir = "./default_images"
    os.makedirs(default_dir, exist_ok=True)
    file_path = os.path.join(default_dir, file.filename)
    
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    return {"message": f"Image {file.filename} added successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)