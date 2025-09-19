import os
import json
import openai
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

openai.api_key = OPENAI_API_KEY


def get_opole_recommendations():
    instructions="""
    You are a travel guide. The user is interested in visiting Opole, a city in Poland. Your task is to provide a list of the places to visit in Opole, specifically considering authentic cultural and historical places. 

    Please follow these steps for the output:
    1. Provide a ranked list of 5 places to visit in Opole, starting with the most significant cultural and historical perspective.
    2. For each place, include:
       - Name of the place.
       - A brief description (1-2 sentences) highlighting its absolute relevance to history, or cultural value.
       - An optional suggestion for an AR or VR experience if available.
       - A rating or ranking score based on historical significance, cultural relevance or user reviews.
    3. Ensure the output is in a structured format, easy to read, and clear for the user. Use headings for each place with relevant details listed below.
    4. The recommendations should reflect a strong emphasis on hidden gems in opole.
    5. Make sure the description is authentic and collected from relevant sources only (Wikipedia, TripAdvisor, UNESCO, EUROPA DATA) as references for the details.
    6. If available, offer a brief recommendation for each place with actions such as "Explore AR tour" or "Read more reviews.
    7. Don't search irrelevant websites for Data.
    8. Make sure under any circumstances you do not use any other websites other than the ones mentioned!!!
    9. You should only return a JSON array, with objects containing:
           - "name"
           - "description" (in bullet points).
    """
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": "I want to go to Opole"}
        ],
        response_format={"type": "json_object"}
    )

    result = response.choices[0].message.content
    return json.loads(result)
