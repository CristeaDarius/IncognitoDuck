import os
import json
import openai
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

openai.api_key = OPENAI_API_KEY


def get_recommendations(place: str, category: str):
    culture_instructions = f"""
         You are a cultural guide for a place called {place}. The user is interested in learning about the culture of {place}. 
         The user only wants the name and a description of places related to the prompt. You should only return a JSON array, 
         with objects containing:
           - "name"
           - "description" (in bullet points).
         
         What "culture" means here:
         - Local food: famous dishes, beverages, and any food-related customs.
         - Traditional dress code: fashion, clothing styles, and significance of attire.
         - Social behavior: people’s behavior, customs, and common social norms.
         - Music, dance, and festivals: popular forms of entertainment, cultural performances, and annual celebrations.
         - Unique cultural events or traditions: any special ceremonies, rituals, or festivals that are significant to the culture of {place}.
    """

    history_instructions = f"""
         You are a historical guide for a place called {place}. The user is interested in learning about the history of {place}. 
         The user only wants the name and a description of places related to the prompt. You should only return a JSON array, 
         with objects containing:
           - "name"
           - "description" (in bullet points).
         
         What "history" means here:
         - Interesting facts about ancient buildings, statues, and landmarks in {place}, highlighting their historical significance.
         - Famous historical figures from {place} and their contributions to both local and global history.
         - The evolution of {place} through different time periods.
         - Major events that shaped the history of {place}.
         - Key historical museums, landmarks, or sites to visit.
    """

    if category == "culture":
        instructions = culture_instructions
    elif category == "history":
        instructions = history_instructions
    else:
        return {"error": "Invalid category. Please choose either 'culture' or 'history'."}

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"I want to go to {place}"}
        ],
        response_format={"type": "json_object"}
    )

    result = response.choices[0].message.content
    return json.loads(result)
