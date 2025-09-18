from fastapi import FastAPI, Query, HTTPException, Request, UploadFile, File
from starlette.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Literal
import os
from pathlib import Path
import shutil
import uuid
from typing import List
import hashlib

# For image search
import chromadb
from PIL import Image
from io import BytesIO
from chromadb.errors import NotFoundError
from transformers import CLIPProcessor, CLIPModel
import torch


app = FastAPI()

#For viewing the images
app.mount("/static", StaticFiles(directory="places"), name="static")

### Images vector database
project_root = Path(__file__).parent
db_path = project_root / "chroma_db"
db_path.mkdir(parents=True, exist_ok=True)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
client = chromadb.PersistentClient(path=str(db_path))
try:
    collection = client.get_collection("images")
except NotFoundError:
    collection = client.create_collection("images", metadata={"hnsw:space": "cosine"})


#### The barebones idea ####
## In the places folder we ll have the list of places
## In each different place, we ll have a description, description.txt
## In info.text, the description should be biased, it will mention if the place
# has a more cultural impact or more of a historical impact

@app.get("/")
async def root():
    return FileResponse("index.html")


@app.get("/recommendations")
async def our_recommendation(
        context: Literal["historical", "cultural"] = Query(..., description="Context"),
):
    filename = f"recommendations_{context}.txt"
    filepath = os.path.join(filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"No recommendations found for context '{context}'")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    return {"recommendations": lines}


# @app.get("/text")
# async def text(
#         text_query: str,
# ):
#     # The model will read the text prompt, then based of the descriptions and feedback,
#     # it will return a place name
#
#

@app.post("/identify")
async def identify(file: UploadFile = File(...)):

    print(collection)

    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error opening image: {str(e)}")

    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    embedding = embeddings.cpu().numpy()

    result = collection.query(
        query_embeddings=embedding,
        n_results=1
    )

    if not result['ids'] or not result['ids'][0]:
        raise HTTPException(status_code=404, detail="No matching image found")

    return {
        "id": result['ids'][0][0],
        "metadata": result['metadatas'][0][0] if result.get('metadatas') else None,
        "distance": result['distances'][0][0]
    }


@app.get("/details")
async def detailed(name: str, request: Request):
    base_dir = os.path.join("places", name)

    if not os.path.isdir(base_dir):
        raise HTTPException(status_code=404, detail=f"Place '{name}' not found")

    desc_path = os.path.join(base_dir, "description.txt")
    description = None
    if os.path.exists(desc_path):
        with open(desc_path, "r", encoding="utf-8") as f:
            description = f.read().strip()

    images_dir = os.path.join(base_dir, "images")
    images = []
    if os.path.isdir(images_dir):
        for file in os.listdir(images_dir):
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                images.append(str(request.url_for("static", path=f"{name}/images/{file}")))

    return {
        "name": name,
        "description": description,
        "images": images,
    }

@app.get("/places_list")
async def places_list():
    base_dir = "places"
    if not os.path.isdir(base_dir):
        raise HTTPException(status_code=404, detail="Places directory not found")

    folders = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]
    return {"places": folders}


from fastapi import FastAPI, Query, HTTPException, Request, UploadFile, File
from starlette.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Literal
import os
from pathlib import Path
import shutil
import uuid
from typing import List
import hashlib

# For image search
import chromadb
from PIL import Image
from io import BytesIO
from chromadb.errors import NotFoundError
from transformers import CLIPProcessor, CLIPModel
import torch


app = FastAPI()

#For viewing the images
app.mount("/static", StaticFiles(directory="places"), name="static")

### Images vector database
project_root = Path(__file__).parent
db_path = project_root / "chroma_db"
db_path.mkdir(parents=True, exist_ok=True)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
client = chromadb.PersistentClient(path=str(db_path))
try:
    collection = client.get_collection("images")
except NotFoundError:
    collection = client.create_collection("images", metadata={"hnsw:space": "cosine"})


#### The barebones idea ####
## In the places folder we ll have the list of places
## In each different place, we ll have a description, description.txt
## In info.text, the description should be biased, it will mention if the place
# has a more cultural impact or more of a historical impact

@app.get("/")
async def root():
    return FileResponse("index.html")


@app.get("/recommendation")
async def our_recommendation(
        context: Literal["historical", "cultural"] = Query(..., description="Context"),
):
    filename = f"recommendations_{context}.txt"
    filepath = os.path.join(filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"No recommendations found for context '{context}'")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    return {"recommendations": lines}


# @app.get("/text")
# async def text(
#         text_query: str,
# ):
#     # The model will read the text prompt, then based of the descriptions and feedback,
#     # it will return a place name
#
#

@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error opening image: {str(e)}")

    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    embedding = embeddings.cpu().numpy()

    result = collection.query(
        query_embeddings=embedding,
        n_results=1,
    )

    if not result['ids'] or not result['ids'][0]:
        raise HTTPException(status_code=404, detail="No matching image found")

    match_id = result['ids'][0][0]
    distance = result['distances'][0][0]
    similarity = 1 - distance

    if similarity < 0.75:
        raise HTTPException(status_code=404, detail=f"No match within threshold. Similarity={similarity:.4f}")

    return {
        "id": match_id,
        "similaritu": similarity
    }


@app.get("/details")
async def detailed(name: str, request: Request):
    base_dir = os.path.join("places", name)

    if not os.path.isdir(base_dir):
        raise HTTPException(status_code=404, detail=f"Place '{name}' not found")

    desc_path = os.path.join(base_dir, "description.txt")
    description = None
    if os.path.exists(desc_path):
        with open(desc_path, "r", encoding="utf-8") as f:
            description = f.read().strip()

    images_dir = os.path.join(base_dir, "images")
    images = []
    if os.path.isdir(images_dir):
        for file in os.listdir(images_dir):
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                images.append(str(request.url_for("static", path=f"{name}/images/{file}")))

    return {
        "name": name,
        "description": description,
        "images": images,
    }

@app.get("/places_list")
async def places_list():
    base_dir = "places"
    if not os.path.isdir(base_dir):
        raise HTTPException(status_code=404, detail="Places directory not found")

    folders = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]
    return {"places": folders}


@app.post("/upload_images")
async def upload_images(place: str = Query(...), files: List[UploadFile] = File(...)):
    base_dir = Path("places") / place
    images_dir = base_dir / "images"

    if not base_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Place '{place}' not found")

    images_dir.mkdir(parents=True, exist_ok=True)

    existing_hashes = set()
    for f in images_dir.iterdir():
        if f.is_file():
            with open(f, "rb") as file_obj:
                existing_hashes.add(hashlib.sha256(file_obj.read()).hexdigest())

    saved_files = []
    skipped_files = []

    for file in files:
        content = await file.read()
        file_hash = hashlib.sha256(content).hexdigest()

        if file_hash in existing_hashes:
            skipped_files.append(file.filename)
            continue

        ext = file.filename.split(".")[-1]
        unique_name = f"{place}_{uuid.uuid4().hex}.{ext}"
        image_path = images_dir / unique_name

        try:
            image = Image.open(BytesIO(content)).convert("RGB")
            inputs = processor(images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                embeddings = model.get_image_features(**inputs)
            embedding = embeddings.cpu().numpy().tolist()

            with open(image_path, "wb") as buffer:
                buffer.write(content)

            collection.add(
                ids=[unique_name],
                embeddings=embedding,
                metadatas=[{"place": place}],
                documents=[f"Image of {place}"]
            )

            saved_files.append(unique_name)
            existing_hashes.add(file_hash)

        except Exception as e:
            if image_path.exists():
                image_path.unlink()
            print(f"Skipped {file.filename}, failed embedding/DB insert: {e}")
            skipped_files.append(file.filename)

    message = f"{len(saved_files)} image(s) uploaded successfully for place '{place}'."
    if skipped_files:
        message += f" {len(skipped_files)} image(s) were skipped (already exist or failed to process)."

    return {
        "message": message,
        "filenames": saved_files,
        "skipped_files": skipped_files
    }
