import os
import openai
from agents import Agent, Runner

# Hardcode the API key
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxx"

# Set the OpenAI API key explicitly for the agent library
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # Set it as an environment variable
openai.api_key = OPENAI_API_KEY  # You may still need to explicitly set it as you did


# Define the function to get recommendations based on the user input (place and choice)
def get_recommendations(place, category):
    # Instructions for culture
    culture_instructions = f"""
         You are a cultural guide for a place called {place}. The user is interested in learning about the culture of {place}. The user onkly wants the name and a description of places related to the propmt. You shoul only return a list, with names of locations related to culture, and their descriptsions. I will explain what i mean when i say culture::
         - Local food: famous dishes, beverages, and any food-related customs.
         - Traditional dress code: fashion, clothing styles, and significance of attire.
         - Social behavior: people’s behavior, customs, and common social norms.
         - Music, dance, and festivals: popular forms of entertainment, cultural performances, and annual celebrations.
         - Unique cultural events or traditions: any special ceremonies, rituals, or festivals that are significant to the culture of {place}.
        Please make sure you return alist, with names and descriptions, and why they are chosen.
        And give the results in array format and the description should be in bullet in points.
    """

    # Instructions for history
    history_instructions = f"""
         You are a historical guide for a place called {place}. The user is interested in learning about the history of {place}. The user  wants the name and a description of places related to the propmt. You shoul only return a list, with names of locations related to history, and their descriptsions. I will explain what i mean when i say historical aspects such as:
        - Interesting facts about ancient buildings, statues, and landmarks in {place}, highlighting their historical significance.
        - Famous historical figures from {place} and their contributions to both local and global history.
        - The evolution of {place} through different time periods (e.g., medieval, Renaissance, modern era), including significant societal and political changes.
        - Major events that shaped the history of {place}, focusing on turning points in its development.
        - Key historical museums, landmarks, or sites that history enthusiasts should visit to deepen their understanding of {place}'s past.
        Please ensure that your descriptions provide a clear and engaging narrative about the history of {place}, offering a thorough exploration of its historical journey.
        And give the results in array format and the description should be in bullet in points.
    """

    # Select the appropriate instructions based on the category choice
    if category == "culture":
        instructions = culture_instructions
    elif category == "history":
        instructions = history_instructions
    else:
        return "Invalid category. Please choose either 'culture' or 'history'."

    # Initialize the agent with dynamic instructions
    agent = Agent(
        name="RecommendationAgent",
        instructions=instructions,
    )

    # Run the query
    result = Runner.run_sync(agent, f"I want to go to {place}")
    return result.final_output


# Main program to accept user input
def main():
    # Get place input from the user
    place = input("Enter the place you want to Explore: ")

    # Ask the user to choose the category
    print("\nChoose a category by number:")
    print("1. Culture")
    print("2. History")

    category_choice = input("\nEnter the number for your choice (1 or 2): ")

    # Convert input to the appropriate category
    if category_choice == "1":
        category = "culture"
    elif category_choice == "2":
        category = "history"
    else:
        print("Invalid choice. Please choose 1 for culture or 2 for history.")
        return

    # Get recommendations based on the place and selected category
    recommendations = get_recommendations(place, category)
    print("\nRecommendations:\n")
    print(recommendations)


if __name__ == "__main__":
    main()
