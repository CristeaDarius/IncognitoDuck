import os
import openai
from agents import Agent, Runner

# Hardcode the API key
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxx"

# Set the OpenAI API key explicitly for the agent library
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # Set it as an environment variable
openai.api_key = OPENAI_API_KEY  # You may still need to explicitly set it as you did

# Initialize the agent
agent = Agent(
    name="RecommendationAgent",
    instructions="""
    You are a travel guide. The user is interested in visiting Opole, a city in Poland. Your task is to provide a list of the best places to visit in Opole, taking into account historical, cultural, and popular landmarks. 

    Please follow these steps for the output:
    1. Provide a ranked list of 5 places to visit in Opole, starting with the most significant or popular.
    2. For each place, include:
       - Name of the place.
       - A brief description (1-2 sentences) highlighting its importance, history, or cultural value.
       - An optional suggestion for an AR or VR experience if available.
       - A rating or ranking score based on popularity, historical significance, or user reviews.
    3. Ensure the output is in a structured format, easy to read, and clear for the user. Use headings for each place with relevant details listed below.
    4. The recommendations should reflect a mix of well-known tourist spots and hidden gems in Opole.
    5. Make sure the description is informative but concise. Include relevant sources (Wikipedia, TripAdvisor, UNESCO, EUROPA DATA) as references for the details.
    6. If available, offer a brief recommendation for each place with actions such as "Explore AR tour" or "Read more reviews.
    7. Don't search irrelevant websites for Data
    8. Make sure under any circumstances you do not use any other websites other than he ones mentioned!!!
    9. "
    """,
)

# Run the query
result = Runner.run_sync(agent, "I want to go to Opole")
# Print the result
print(result.final_output)
