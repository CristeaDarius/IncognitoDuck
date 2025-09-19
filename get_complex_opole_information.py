import os
import json
import openai
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

openai.api_key = OPENAI_API_KEY


def get_place_list(city: str):
    instructions = f"""
    You are a cultural and historical researcher. 
    The user is interested in {city}, Poland. 

    Step 1: Identify **all known cultural and historical places** in {city}, Poland. 
    Include landmarks, monuments, churches, museums, parks, squares, theaters, towers, bridges, historic houses, cemeteries, statues, and any site with cultural or historical value. 
    Do not limit yourself to just the "most important" or "top" ones.
    
    Step 2: Return ONLY a JSON array of strings, where each string is the exact name of one place.
    Step 3: Do not provide descriptions yet. Just names.

    Sources: Wikipedia, TripAdvisor, UNESCO, EUROPA DATA.
    You must only use those sources.
    """

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"List ALL cultural and historical places in {city} as JSON array of names only."}
        ],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)


def get_all_places(city: str, max_rounds: int = 5):
    """Fetch places in multiple rounds until no new ones are found."""
    all_places = []
    seen = set()

    for i in range(max_rounds):
        print(f"🔎 Fetching round {i+1} ...")
        response = get_place_list(city)

        # handle either {"places": [...]} or just a list
        places = response.get("places", response)

        new_places = [p for p in places if p not in seen]
        if not new_places:
            print("✅ No new places found, stopping.")
            break

        print(f"   ➕ Found {len(new_places)} new places")
        all_places.extend(new_places)
        seen.update(new_places)

    return all_places


def get_detailed_description(city: str, place: str):
    instructions = f"""
    You are a cultural and historical researcher. 
    The user is interested in {city}, Poland. 

    Provide a long, detailed, aggregated description of the place: "{place}". 
    Use trusted sources only: Wikipedia, TripAdvisor, UNESCO, EUROPA DATA.
    You must only use those sources.

    Your response must be returned as JSON with:
    - "name": the place name
    - "detailed_description": several paragraphs combining:
        * Historical background
        * Cultural significance
        * Unique facts and hidden gems
        * Notable events or people
        * Visitor tips
    - "sources": which of [Wikipedia, TripAdvisor, UNESCO, EUROPA DATA] were relevant
    """

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"Give me a detailed description of {place} in {city} as JSON."}
        ],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)


def save_place_description(city: str, place_data: dict):
    base_dir = os.path.join("places", city)
    os.makedirs(base_dir, exist_ok=True)

    dir_name = place_data["name"].lower().replace(" ", "_")
    place_dir = os.path.join(base_dir, dir_name)
    os.makedirs(place_dir, exist_ok=True)

    desc_path = os.path.join(place_dir, "description.txt")
    with open(desc_path, "w", encoding="utf-8") as f:
        f.write(place_data["detailed_description"])
        f.write("\n\n---\nSources: " + ", ".join(place_data.get("sources", [])))


def get_all_opole_places():
    city = "Opole"
    places = get_all_places(city)
    detailed_places = []

    for place in places:
        print(f"📖 Fetching details for {place} ...")
        details = get_detailed_description(city, place)
        detailed_places.append(details)

        save_place_description(city, details)

    return detailed_places


if __name__ == "__main__":
    city = "Opole"

    # Just get the complete place list (multi-round fetch)
    places = get_all_places(city)

    print("\n✅ Full place list fetched successfully!")
    print(json.dumps(places, indent=2, ensure_ascii=False))

    results = get_all_opole_places()
    print("\n✅ Finished fetching all places with descriptions! Saved to places/Opole/")
    print(json.dumps(results, indent=2, ensure_ascii=False))
