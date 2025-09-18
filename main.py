from fastapi import FastAPI, Query, HTTPException, Request
from starlette.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Literal
import os
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from transformers import CLIPProcessor, CLIPModel


app = FastAPI()

#For viewing the images
app.mount("/static", StaticFiles(directory="places"), name="static")

### Images vector database
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
client = chromadb.PersistentClient(path="../chroma_db")
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

@app.get("/identify")
async def identify(
        file: UploadFile = File(...),
):
    # The object/location will be identified, based on the images we have in the database
    #it will return a place name

    #get the embeddings of that image
    image_embedding = []
    collection.query(
        query_embeddings=image_embedding,
        n_results=1
    )


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
