import streamlit as st
from langchain.chat_models import ChatOllama
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
import re
from datetime import datetime, timedelta
import pandas as pd

# Load environment variables
load_dotenv()

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
    st.session_state.pantry = []  # Just store ingredient names now
    st.session_state.dietary_needs = ""
    st.session_state.meal_type = ""
    st.session_state.current_recipes = []  # Store the last shown recipes

# Shelf Life Parsing Functions
def parse_shelf_life(shelf_life_str):
    """Parse shelf life string into days"""

    # No expiration
    if not shelf_life_str or pd.isna(shelf_life_str):
        return float('inf')  
    
    # Extract numbers and unit
    numbers = re.findall(r"(\d+\.?\d*)", shelf_life_str)
    unit = re.search(r"(Day|Week|Month|Year)", shelf_life_str, re.IGNORECASE)
    
    if not numbers or not unit:
        return float('inf')
    
    avg_value = (float(numbers[0]) + (float(numbers[-1]) if len(numbers) > 1 else float(numbers[0]))) / 2
    unit = unit.group(1).lower()
    
    # Convert to days
    if unit == 'day':
        return avg_value
    elif unit == 'week':
        return avg_value * 7
    elif unit == 'month':
        return avg_value * 30
    elif unit == 'year':
        return avg_value * 365
    return float('inf')

# Neo4j Recipe Recommendation function
def get_recipe_recommendations(ingredient_names, dietary_needs="", meal_type=""):
    with driver.session() as session:
        # get all shelf life data for the ingredients
        shelf_life_data = session.run("""
            UNWIND $ingredients AS ingredient
            MATCH (i:Ingredient)
            WHERE toLower(trim(i.name)) = toLower(trim(ingredient))
            RETURN i.name AS name, i.shelf_life AS shelf_life
            """, ingredients=[x.lower() for x in ingredient_names])
        
        # process shelf life data
        shelf_life_map = {}
        for record in shelf_life_data:
            name = record["name"]
            shelf_life = record["shelf_life"]
            days = parse_shelf_life(shelf_life)
            shelf_life_map[name.lower()] = {
                "shelf_life": shelf_life,
                "days_remaining": days
            }
        
        # find matching recipes acc to the constraints
        result = session.run("""
            // Converting input to list of lowercase strings
            WITH [x IN $ingredients | toLower(trim(x))] AS searchIngredients
            
            // Find recipes using these ingredients
            MATCH (r:Recipe)-[:USES]->(i:Ingredient)
            WHERE toLower(trim(i.name)) IN searchIngredients
            
            // Group by recipe and count matches
            WITH r, searchIngredients, collect(i) AS usedIngredients, 
                 size([(r)-[:USES]->(x) | x]) AS totalIngredients
                 
            // TODO: meal type filter - check if this works
            WHERE ($meal_type = "" OR toLower(r.meal_type) = toLower($meal_type))
            
            // Find missing ingredients
            OPTIONAL MATCH (r)-[:USES]->(missing:Ingredient)
            WHERE NOT toLower(trim(missing.name)) IN searchIngredients
            
            RETURN r.title AS title,
                   [(r)-[:USES]->(i) | i.name] AS ingredients,
                   size(usedIngredients) AS matches,
                   totalIngredients,
                   collect(missing.name) AS missing_ingredients,
                   r.directions AS steps
            ORDER BY size(usedIngredients) DESC, totalIngredients ASC
            LIMIT 20
            """, 
            ingredients=[x.lower() for x in ingredient_names],
            meal_type=meal_type.lower()
        )
        
        recipes = []
        for record in result:
            recipe = dict(record)
            recipe["coverage"] = recipe["matches"] / float(recipe["totalIngredients"])
            
            # Calculate urgency information (the ingredients with shorter life span have a higher score)
            urgent_ingredients = []
            urgency_score = 0
            for ing_name in recipe["ingredients"]:
                ing_data = shelf_life_map.get(ing_name.lower())
                if ing_data and ing_data["days_remaining"] < 7:
                    urgent_ingredients.append(f"{ing_name} ({ing_data['days_remaining']} days)")
                if ing_data:
                    urgency_score += 1.0 / (ing_data["days_remaining"] + 1)
            
            recipe["urgent_ingredients"] = urgent_ingredients
            recipe["urgency_score"] = urgency_score
            recipes.append(recipe)
        
        # Sort by urgency score (highest appears first) 
        recipes.sort(key=lambda x: (-x["urgency_score"], -x["coverage"]))
        return recipes

# Initialize the LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0.7,
    streaming=True
)

# Initialize the prompt template
recipe_prompt = PromptTemplate(
    input_variables=["ingredients", "dietary_needs", "meal_type", "recipe_results"],
    template="""
    You're a professional chef assistant helping reduce food waste. Recommend recipes prioritizing ingredients that will expire soon.
    
    Available ingredients:
    {ingredients}
    
    Dietary preferences: {dietary_needs}
    Meal type: {meal_type}
    
    Top matching recipes from database (indicates soon-to-expire ingredients):
    {recipe_results}
    
    Format your response with:
    For each recipe:
    1. **Recipe Name** (Match Percentage)
       - Uses: [list of matching ingredients]
       - Missing: [list of missing ingredients if any]
       - Why Recommended: [1-2 sentence explanation]
       - Quick Instructions
    
    Include a final tip about ingredient substitutions.
    """
)
# Streamlit UI code
st.set_page_config(layout="wide")
st.title("🍳 Pantry to Plate - Recipe Recommender")
st.info("""This chatbot interface is designed to help you find recipes 
that would prioritize the ingredients you have in your pantry.""")

with st.sidebar:
    st.sidebar.header("How this works")
    st.markdown("""
    1. Add ingredients you have
    2. Set your dietary preferences
    3. Get smart recommendations that prioritize:
       - Using ingredients that expire soonest
       - Maximizing ingredients you have
    """)
    st.sidebar.header(""" Example Questions - 
    Recommend recipes based on my provided ingredients and preferences""")
    st.divider()
    
    st.subheader("Your Pantry")
    ingredient_name = st.text_input("Add ingredients (comma separated)")
    if st.button("Update Pantry") and ingredient_name:
        st.session_state.pantry = [x.strip() for x in ingredient_name.split(",")]
        st.rerun()
    
    # Display current pantry
    if st.session_state.get('pantry'):
        st.subheader("Current Ingredients")
        for item in st.session_state.pantry:
            st.markdown(f"- {item}")
    
    st.subheader("Preferences")
    st.session_state.dietary_needs = st.text_input("Dietary needs")
    st.session_state.meal_type = st.selectbox(
        "Meal type",
        ["Any", "Breakfast", "Lunch", "Dinner", "Snack"]
    )

# Chat interface
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask for recipe recommendations"):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Determine if this is a recipe request
    is_recipe_request = any(keyword in prompt.lower() for keyword in ["recipe", "cook", "make", "suggest", "recommend"])

    with st.chat_message("assistant"):
        if is_recipe_request and st.session_state.get('pantry'):
            # Get recommendations from knowledge graph
            recipes = get_recipe_recommendations(
                st.session_state.pantry,
                st.session_state.dietary_needs,
                st.session_state.meal_type
            )
            
            # Storing the recipes in the session
            st.session_state.current_recipes = recipes

            # Format results with urgency info
            recipe_results = []
            for r in recipes:
                urgent_ings = [
                    ing['name'] for ing in r['ingredient_details'] 
                    if ing.get('days_remaining', float('inf')) < 7
                ]
                
                recipe_results.append(
                    f"**{r['title']}** ({r['coverage']:.0%} match)\n"
                    f"- Uses: {', '.join(r['ingredients'])}\n"
                    f"{'- URGENT: ' + ', '.join(urgent_ings) if urgent_ings else ''}\n"
                    f"{'- Needs: ' + ', '.join(r['missing_ingredients']) if r['missing_ingredients'] else ''}"
                )
            
            # Generate LLM response
            response = llm([
                HumanMessage(content=recipe_prompt.format(
                    ingredients=", ".join(st.session_state.pantry),
                    dietary_needs=st.session_state.dietary_needs or "none",
                    meal_type=st.session_state.meal_type or "any",
                    recipe_results="\n\n".join(recipe_results)
                ))
            ])

            full_response = response.content if hasattr(response, 'content') else str(response)

            
        else:
            # Regular chat response BUT with recipe context if available
            if st.session_state.get('current_recipes'):
                # Add recipe context to general questions
                recipe_titles = [r['title'] for r in st.session_state.current_recipes]
                recipe_context = f"Recent recipes shown: {', '.join(recipe_titles)}"
                enhanced_prompt = f"{prompt}\n\nContext: {recipe_context}"
            else:
                enhanced_prompt = prompt
                
            response_container = st.empty()
            full_response = ""
            
            if "conversation_chain" not in st.session_state:
                st.session_state.conversation_chain = LLMChain(
                    llm=llm,
                    prompt=PromptTemplate(
                        input_variables=["history", "human_input"],
                        template="{history}\nUser: {human_input}\nAssistant:"
                    ),
                    memory=st.session_state.memory
                )
            
            for chunk in st.session_state.conversation_chain.stream({"human_input": enhanced_prompt}):
                if isinstance(chunk, dict) and "text" in chunk:
                    text_chunk = chunk["text"]
                    full_response += text_chunk
                    response_container.markdown(full_response)
        
        st.markdown(full_response)
        
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                # st.markdown(response.content)
                # st.session_state.chat_history.append({"role": "assistant", "content": response.content})
        # else:
        #     st.warning("Please add ingredients to your pantry first")
        #     st.session_state.chat_history.append({"role": "assistant", "content": "Please add ingredients to your pantry first"})