from fastapi import FastAPI, Query, HTTPException, Request, UploadFile, File
from starlette.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Literal, Optional
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

MODERATION_PASSWORD = "supersecret123"

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
#     # The model will read the text prompt, then based of the descriptions and feedback_texts,
#     # it will return a place name
#
#

@app.post("/identify")
async def identify(file: UploadFile = File(...)): # TO RETURN THE PLACE NAME NOT EMBEDDING ID
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
#     # The model will read the text prompt, then based of the descriptions and feedback_texts,
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
    place_name = "_".join(match_id.split("_")[:-1])

    if similarity < 0.75:
        raise HTTPException(status_code=404, detail=f"No match within threshold. Similarity={similarity:.4f}")

    return {
        "image_id": match_id,
        "name": place_name,
        "similarity": similarity
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

    videos_dir = os.path.join(base_dir, "videos")
    videos = []
    if os.path.isdir(videos_dir):
        for file in os.listdir(videos_dir):
            if file.lower().endswith((".mp4", ".mov", ".avi", ".webm")):
                videos.append(str(request.url_for("static", path=f"{name}/videos/{file}")))

    feedback_dir = os.path.join(base_dir, "feedback_texts")
    feedback_texts = []
    if os.path.isdir(feedback_dir):
        for file in os.listdir(feedback_dir):
            if file.lower().endswith(".txt"):
                with open(os.path.join(feedback_dir, file), "r", encoding="utf-8") as f:
                    feedback_texts.append(f.read().strip())

    return {
        "name": name,
        "description": description,
        "images": images,
        "videos": videos,
        "feedback_texts": feedback_texts,
    }

@app.get("/places_list")
async def places_list():
    base_dir = "places"
    if not os.path.isdir(base_dir):
        raise HTTPException(status_code=404, detail="Places directory not found")

    folders = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]
    return {"places": folders}

@app.post("/post_feedback")
async def post_feedback(
        place: Optional[str] = Query(None),
        files: Optional[List[UploadFile]] = None,
        text_feedback: Optional[str] = Query(None)
):
    if not place:
        raise HTTPException(status_code=400, detail="Parameter 'place' is required")

    unmod_root = Path("unmoderated_places")
    place_dir = None

    # Search for the place folder in unmoderated_places
    for subdir in unmod_root.rglob("*"):
        if subdir.is_dir() and subdir.name == place:
            place_dir = subdir
            break

    # If not found, copy from places folder
    if not place_dir:
        places_root = Path("places")
        source_place_dir = None

        for country_dir in places_root.iterdir():
            if not country_dir.is_dir():
                continue
            candidate = country_dir / place
            if candidate.is_dir():
                source_place_dir = candidate
                break

        if not source_place_dir:
            raise HTTPException(status_code=404, detail=f"Place '{place}' not found in either unmoderated_places or places")

        # Determine the same subdirectory name (country) in unmoderated_places
        country_subdir = unmod_root / source_place_dir.parent.name
        country_subdir.mkdir(parents=True, exist_ok=True)

        # Copy the place folder
        place_dir = country_subdir / place
        shutil.copytree(source_place_dir, place_dir)

    # Create necessary subdirectories
    images_dir = place_dir / "images"
    videos_dir = place_dir / "videos"
    feedback_dir = place_dir / "feedback_texts"

    images_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    feedback_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    skipped_files = []

    # Handle files if provided
    if files is not None:
        existing_hashes_images = {hashlib.sha256(f.read_bytes()).hexdigest() for f in images_dir.glob("*") if f.is_file()}
        existing_hashes_videos = {hashlib.sha256(f.read_bytes()).hexdigest() for f in videos_dir.glob("*") if f.is_file()}

        for file in files:
            content = await file.read()
            ext = file.filename.split(".")[-1].lower()
            unique_name = f"{place}_{uuid.uuid4().hex}.{ext}"

            try:
                if ext in ["png", "jpg", "jpeg", "gif", "webp"]:
                    file_hash = hashlib.sha256(content).hexdigest()
                    if file_hash in existing_hashes_images:
                        skipped_files.append(file.filename)
                        continue
                    Image.open(BytesIO(content)).verify()
                    path = images_dir / unique_name
                    with open(path, "wb") as f_out:
                        f_out.write(content)
                    existing_hashes_images.add(file_hash)

                elif ext in ["mp4", "mov", "avi", "webm"]:
                    file_hash = hashlib.sha256(content).hexdigest()
                    if file_hash in existing_hashes_videos:
                        skipped_files.append(file.filename)
                        continue
                    path = videos_dir / unique_name
                    with open(path, "wb") as f_out:
                        f_out.write(content)
                    existing_hashes_videos.add(file_hash)

                else:
                    skipped_files.append(file.filename)
                    continue

                saved_files.append(unique_name)

            except Exception as e:
                print(f"Skipped {file.filename}: {e}")
                skipped_files.append(file.filename)

    # Save text feedback if provided
    feedback_file = None
    if text_feedback:
        feedback_file = feedback_dir / f"{place}_{uuid.uuid4().hex}.txt"
        with open(feedback_file, "w", encoding="utf-8") as f:
            f.write(text_feedback)

    message = f"{len(saved_files)} file(s) uploaded successfully for place '{place}'."
    if skipped_files:
        message += f" {len(skipped_files)} file(s) were skipped (duplicate or invalid type)."
    if feedback_file:
        message += f" Feedback saved as {feedback_file.name}."

    return {
        "message": message,
        "files_saved": saved_files,
        "skipped_files": skipped_files,
        "feedback_file": feedback_file.name if feedback_file else None
    }



@app.get("/get_all_unmoderated")
async def get_all_unmoderated():
    base_root = Path("unmoderated_places")
    all_places = []

    if not base_root.is_dir():
        return {"places": []}

    for country_dir in base_root.iterdir():  # Skip top-level directories (city name)
        if not country_dir.is_dir():
            continue

        for place_dir in country_dir.iterdir():
            if not place_dir.is_dir():
                continue

            place_name = place_dir.name

            images_dir = place_dir / "images"
            videos_dir = place_dir / "videos"
            feedback_dir = place_dir / "feedback_texts"

            images = []
            if images_dir.is_dir():
                for file in images_dir.iterdir():
                    if file.is_file() and file.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
                        images.append({
                            "name_of_image": file.name,
                            "url": str(file)
                        })

            videos = []
            if videos_dir.is_dir():
                for file in videos_dir.iterdir():
                    if file.is_file() and file.suffix.lower() in [".mp4", ".mov", ".avi", ".webm"]:
                        videos.append({
                            "name_of_video": file.name,
                            "url": str(file)
                        })

            # Collect feedback texts
            feedback_texts = []
            if feedback_dir.is_dir():
                for file in feedback_dir.iterdir():
                    if file.is_file() and file.suffix.lower() == ".txt":
                        with open(file, "r", encoding="utf-8") as f:
                            feedback_texts.append(f.read().strip())

            all_places.append({
                "place": place_name,
                "images": images,
                "videos": videos,
                "feedback_texts": feedback_texts
            })

    return {"places": all_places}


def check_password(password: str):
    if password != MODERATION_PASSWORD:
        raise HTTPException(status_code=403, detail="Invalid password")


def find_place_dir(base_root: Path, place_name: str) -> Path:
    for subdir in base_root.rglob("*"):  # recursively searches all subdirectories
        if subdir.is_dir() and subdir.name == place_name:
            return subdir
    return None


def get_moderated_place_dir(unmoderated_place_dir: Path) -> Path:
    """
    Build the equivalent moderated path inside 'places/',
    preserving the parent structure from 'unmoderated_places'.
    """
    relative_path = unmoderated_place_dir.relative_to("unmoderated_places")
    return Path("places") / relative_path


@app.post("/moderate_feedback_text")
async def moderate_feedback_text(
        place: str = Query(...),
        feedback_file_name: str = Query(...),
        password: str = Query(...)
):
    check_password(password)

    place_dir = find_place_dir(Path("unmoderated_places"), place)
    if not place_dir:
        raise HTTPException(status_code=404, detail=f"Place '{place}' not found in unmoderated_places")

    unmod_feedback_dir = place_dir / "feedback_texts"
    if not unmod_feedback_dir.is_dir():
        raise HTTPException(status_code=404, detail="Unmoderated feedback folder not found")

    feedback_file = unmod_feedback_dir / feedback_file_name
    if not feedback_file.is_file():
        raise HTTPException(status_code=404, detail="Feedback file not found")

    moderated_feedback_dir = get_moderated_place_dir(place_dir) / "feedback_texts"
    moderated_feedback_dir.mkdir(parents=True, exist_ok=True)

    shutil.move(str(feedback_file), moderated_feedback_dir / feedback_file_name)

    return {"message": f"Feedback '{feedback_file_name}' moved to moderated folder"}


@app.post("/moderate_video")
async def moderate_video(
        place: str = Query(...),
        video_file_name: str = Query(...),
        password: str = Query(...)
):
    check_password(password)

    place_dir = find_place_dir(Path("unmoderated_places"), place)
    if not place_dir:
        raise HTTPException(status_code=404, detail=f"Place '{place}' not found in unmoderated_places")

    unmod_video_dir = place_dir / "videos"
    if not unmod_video_dir.is_dir():
        raise HTTPException(status_code=404, detail="Unmoderated videos folder not found")

    video_file = unmod_video_dir / video_file_name
    if not video_file.is_file():
        raise HTTPException(status_code=404, detail="Video file not found")

    moderated_video_dir = get_moderated_place_dir(place_dir) / "videos"
    moderated_video_dir.mkdir(parents=True, exist_ok=True)

    shutil.move(str(video_file), moderated_video_dir / video_file_name)

    return {"message": f"Video '{video_file_name}' moved to moderated folder"}


@app.post("/moderate_image")
async def moderate_image(
        place: str = Query(...),
        image_file_name: str = Query(...),
        password: str = Query(...)
):
    check_password(password)

    place_dir = find_place_dir(Path("unmoderated_places"), place)
    if not place_dir:
        raise HTTPException(status_code=404, detail=f"Place '{place}' not found in unmoderated_places")

    unmod_image_dir = place_dir / "images"
    if not unmod_image_dir.is_dir():
        raise HTTPException(status_code=404, detail="Unmoderated images folder not found")

    image_file = unmod_image_dir / image_file_name
    if not image_file.is_file():
        raise HTTPException(status_code=404, detail="Image file not found")

    moderated_image_dir = get_moderated_place_dir(place_dir) / "images"
    moderated_image_dir.mkdir(parents=True, exist_ok=True)

    shutil.move(str(image_file), moderated_image_dir / image_file_name)

    return {"message": f"Image '{image_file_name}' moved to moderated folder"}
