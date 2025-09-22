from fastapi import FastAPI, Query, HTTPException, Request, UploadFile, File
from starlette.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Literal, Optional
import os
from pathlib import Path
import shutil
import uuid
from typing import List
import hashlib
from dotenv import load_dotenv
import os
from fastapi import Form
import json

# For image search
import chromadb
from PIL import Image
from io import BytesIO
from chromadb.errors import NotFoundError
from transformers import CLIPProcessor, CLIPModel
import torch

from recommend_unpopular_city import get_recommendations
from recommend_opole import get_opole_recommendations


app = FastAPI()

# origins = [
#     "http://127.0.0.1:5500",
#     "http://localhost:5500",
# ]

# Cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
MODERATION_PASSWORD = os.getenv("MODERATION_PASSWORD")

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


@app.get("/")
async def root():
    return FileResponse("index.html")

@app.get("/opole-recommendations")
async def opole_recommendations():
    recs = get_opole_recommendations()
    return {
        "recommendations": recs
    }



TEMP_DIR = "temp_results"
os.makedirs(TEMP_DIR, exist_ok=True)
@app.get("/recommendations-generate")
async def recommendations_generate(
        context: Literal["history", "culture"] = Query(..., description="Context"),
        place: str = Query(..., description="Place name"),
):
    try:
        recs = get_recommendations(place, context)

        temp_filename = f"{uuid.uuid4().hex}.json"
        temp_path = os.path.join(TEMP_DIR, temp_filename)

        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump({
                "place": place,
                "context": context,
                "recommendations": recs,
            }, f, ensure_ascii=False, indent=2)

        return {"file_id": temp_filename}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommendations-result/{file_id}")
async def recommendations_result(file_id: str):
    temp_path = os.path.join(TEMP_DIR, file_id)

    if not os.path.isfile(temp_path):
        raise HTTPException(status_code=404, detail="Result not ready or file not found")

    with open(temp_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

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

    tokens = place_name.split("_")
    half = len(tokens) // 2
    if len(tokens) % 2 == 0 and tokens[:half] == tokens[half:]:
        place_name = "_".join(tokens[:half])

    if similarity < 0.75:
        raise HTTPException(
            status_code=404,
            detail=f"No match within threshold. Similarity={similarity:.4f}"
        )

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

    result = {}
    for city in os.listdir(base_dir):
        city_path = os.path.join(base_dir, city)
        if os.path.isdir(city_path):
            subdirs = [
                name for name in os.listdir(city_path)
                if os.path.isdir(os.path.join(city_path, name))
            ]
            result[city] = {
                "count": len(subdirs),
                "places": subdirs
            }

    return result

@app.post("/post_feedback")
async def post_feedback(
        place: str = Form(...),
        # files: Optional[List[UploadFile]] = None,
        files: Optional[List[UploadFile]] = File(None),
        text_feedback: Optional[str] = Form(None)
):
    if not place:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "missing_parameter",
                "message": "Parameter 'place' is required."
            }
        )

    unmod_root = Path("unmoderated_places")
    place_dir = None

    for subdir in unmod_root.rglob("*"):
        if subdir.is_dir() and subdir.name == place:
            place_dir = subdir
            break

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
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "place_not_found",
                    "message": f"Place '{place}' not found in either unmoderated_places or places."
                }
            )

        country_subdir = unmod_root / source_place_dir.parent.name
        country_subdir.mkdir(parents=True, exist_ok=True)

        place_dir = country_subdir / place
        shutil.copytree(source_place_dir, place_dir)

    images_dir = place_dir / "images"
    videos_dir = place_dir / "videos"
    feedback_dir = place_dir / "feedback_texts"
    for d in (images_dir, videos_dir, feedback_dir):
        d.mkdir(parents=True, exist_ok=True)

    saved_files = []
    file_errors = []

    # Handle files if provided
    if files is not None:
        existing_hashes_images = {
            hashlib.sha256(f.read_bytes()).hexdigest()
            for f in images_dir.glob("*") if f.is_file()
        }
        existing_hashes_videos = {
            hashlib.sha256(f.read_bytes()).hexdigest()
            for f in videos_dir.glob("*") if f.is_file()
        }

        for file in files:
            content = await file.read()
            ext = file.filename.split(".")[-1].lower()
            unique_name = f"{place}_{uuid.uuid4().hex}.{ext}"

            try:
                file_hash = hashlib.sha256(content).hexdigest()

                if ext in ["png", "jpg", "jpeg", "gif", "webp"]:
                    if file_hash in existing_hashes_images:
                        file_errors.append({
                            "filename": file.filename,
                            "reason": "duplicate_image"
                        })
                        continue
                    # Verify image validity
                    try:
                        Image.open(BytesIO(content)).verify()
                    except Exception:
                        file_errors.append({
                            "filename": file.filename,
                            "reason": "corrupted_image"
                        })
                        continue
                    path = images_dir / unique_name
                    with open(path, "wb") as f_out:
                        f_out.write(content)
                    existing_hashes_images.add(file_hash)

                elif ext in ["mp4", "mov", "avi", "webm"]:
                    if file_hash in existing_hashes_videos:
                        file_errors.append({
                            "filename": file.filename,
                            "reason": "duplicate_video"
                        })
                        continue
                    path = videos_dir / unique_name
                    with open(path, "wb") as f_out:
                        f_out.write(content)
                    existing_hashes_videos.add(file_hash)

                else:
                    file_errors.append({
                        "filename": file.filename,
                        "reason": "unsupported_file_type",
                        "allowed_types": ["png", "jpg", "jpeg", "gif", "webp", "mp4", "mov", "avi", "webm"]
                    })
                    continue

                saved_files.append(unique_name)

            except Exception as e:
                file_errors.append({
                    "filename": file.filename,
                    "reason": "internal_error",
                    "details": str(e)
                })

    feedback_file = None
    if text_feedback:
        feedback_file = feedback_dir / f"{place}_{uuid.uuid4().hex}.txt"
        with open(feedback_file, "w", encoding="utf-8") as f:
            f.write(text_feedback)

    return {
        "place": place,
        "summary": {
            "files_saved": len(saved_files),
            "files_failed": len(file_errors),
            "feedback_saved": bool(feedback_file)
        },
        "saved_files": saved_files,
        "file_errors": file_errors,
        "feedback_file": feedback_file.name if feedback_file else None
    }


@app.get("/get_all_unmoderated")
async def get_all_unmoderated():
    base_root = Path("unmoderated_places")
    all_places = []

    if not base_root.is_dir():
        return {"places": []}

    for country_dir in base_root.iterdir():
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

            feedback_texts = []
            if feedback_dir.is_dir():
                for file in feedback_dir.iterdir():
                    if file.is_file() and file.suffix.lower() == ".txt":
                        with open(file, "r", encoding="utf-8") as f:
                            feedback_texts.append({
                                "name_of_feedback": file.name,
                                "content": f.read().strip()
                            })

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
    for subdir in base_root.rglob("*"):
        if subdir.is_dir() and subdir.name == place_name:
            return subdir
    return None


def get_moderated_place_dir(unmoderated_place_dir: Path) -> Path:
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
    target_path = moderated_image_dir / image_file_name
    shutil.move(str(image_file), target_path)

    try:
        image = Image.open(target_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            embeddings = model.get_image_features(**inputs)

        embeddings = embeddings.cpu().numpy().tolist()

        collection.add(
            ids=[f"{place}"],
            embeddings=embeddings,
            metadatas=[{"place": place, "file_name": image_file_name}],
            documents=[f"Image of {place}: {image_file_name}"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add image to ChromaDB: {str(e)}")

    return {"message": f"Image '{image_file_name}' moved to moderated folder and added to ChromaDB"}