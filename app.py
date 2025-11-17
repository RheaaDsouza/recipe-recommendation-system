import os
import re
import pandas as pd

import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from neo4j import GraphDatabase

from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Initialize Neo4j driver only when a URI is provided.
driver = None
if NEO4J_URI:
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    except Exception:
        driver = None

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
    st.session_state.pantry = []  
    st.session_state.dietary_needs = ""
    st.session_state.allergies = ""
    st.session_state.current_recipes = []  # store last shown recipes

# Shelf Life Parsing Function
def parse_shelf_life(shelf_life_str):
    """Parse shelf life string into days - updated for FoodKeeper format"""
    
    # Handle empty strings
    if not shelf_life_str or pd.isna(shelf_life_str) or shelf_life_str == '':
        return float('inf')
    
    shelf_life_str = str(shelf_life_str).lower().strip()
    
    # Handle "No expiration" or similar cases
    if any(term in shelf_life_str for term in ['no expiration', 'indefinite', 'infinite', 'none']):
        return float('inf')
    
    # Handle ranges like "3-5 Days", "1-2 Weeks", etc.
    range_match = re.search(r'(\d+)\s*-\s*(\d+)\s*(day|week|month|year)', shelf_life_str)
    if range_match:
        min_val, max_val, unit = range_match.groups()
        avg_value = (float(min_val) + float(max_val)) / 2
    else:
        # Handle single values like "7 Days", "2 Weeks", etc.
        single_match = re.search(r'(\d+)\s*(day|week|month|year)', shelf_life_str)
        if single_match:
            min_val, unit = single_match.groups()
            avg_value = float(min_val)
        else:
            return float('inf')

    # Convert to days
    unit = unit.lower()
    if unit == 'day':
        return avg_value
    elif unit == 'week':
        return avg_value * 7
    elif unit == 'month':
        return avg_value * 30
    elif unit == 'year':
        return avg_value * 365
    return float('inf')

def normalize_ingredient_data(ingredients):
    """Convert ingredients to consistent format"""
    normalized = []
    for ing in ingredients:
        if isinstance(ing, dict):
            name = (ing['name'] or '').strip()
            category = (ing['category'] or '').strip()
        else:
            name = str(ing).strip()
            category = ''
        if name:
            normalized.append({'name': name, 'category': category})
    return normalized

# Dietary preference mapping (module-level so it can be used in queries/filters)
DIET_MAP = {
    "vegetarian": ["meat", "beef", "pork", "lamb", "goat", "venison", "poultry", "chicken", "turkey", "duck"],
    "vegan": ["meat", "beef", "pork", "lamb", "goat", "venison", "poultry", "chicken", "turkey", "duck", 
              "seafood", "fish", "salmon", "tuna", "shrimp", "prawn", "crab", "lobster", "milk", "cheese", 
              "butter", "cream", "yogurt", "egg", "honey"],
    "gluten-free": ["gluten", "wheat", "barley", "rye", "flour", "pasta", "bread"],
    "pescatarian": ["meat", "beef", "pork", "lamb", "goat", "venison", "poultry", "chicken", "turkey", "duck"],
    "dairy-free": ["milk", "cheese", "butter", "cream", "yogurt", "ghee"],
}

# Allergy/category mapping and checks
ALLERGY_CATEGORY_MAP = {
    "dairy": {"Dairy Products & Eggs"},
    "egg": {"Dairy Products & Eggs"},
    "seafood": {"Seafood"},
    "shellfish": {"Seafood"},
    "fish": {"Seafood"},
    "meat": {"Meat"},
    "poultry": {"Poultry"},
    "nuts": {"Nuts & Seeds", "Shelf Stable Foods"},
    "peanut": {"Nuts & Seeds", "Shelf Stable Foods"},
    "soy": {"Shelf Stable Foods"},
    "wheat": {"Grains & Flours"},
    "gluten": {"Grains & Flours"},
}

ALLERGY_NAME_KEYWORDS = {
    'dairy': ['milk', 'cheese', 'butter', 'cream', 'yogurt', 'ghee', 'whey', 'casein'],
    'egg': ['egg', 'eggs', 'mayonnaise', 'mayo'],
    'nuts': ['almond', 'peanut', 'cashew', 'walnut', 'pecan', 'hazelnut', 'pistachio', 'macadamia', 'brazil'],
    'peanut': ['peanut', 'groundnut'],
    'shellfish': ['shrimp', 'crab', 'lobster', 'prawn', 'scallop', 'clam', 'mussel', 'oyster'],
    'fish': ['fish', 'salmon', 'tuna', 'cod', 'trout', 'bass'],
    'soy': ['soy', 'soya', 'tofu', 'edamame', 'miso', 'tempeh'],
    'wheat': ['wheat', 'flour', 'bread', 'pasta', 'couscous'],
    'gluten': ['wheat', 'barley', 'rye', 'malt', 'brewer'],
}

def violates_allergies(ingredients, allergy_text):
    """Check if ingredient list violates user allergies"""
    if not allergy_text:
        return False

    allergy_text_lower = allergy_text.lower()
    matched_allergies = [k for k in ALLERGY_CATEGORY_MAP.keys() if k in allergy_text_lower]
    
    if not matched_allergies:
        return False

    normalized_ingredients = normalize_ingredient_data(ingredients)
    
    categories = set()
    names_concat = []
    for ing in normalized_ingredients:
        name = ing['name'].lower()
        category = ing['category'].lower()
        names_concat.append(name)
        if category:
            categories.add(category)

    names_text = ' '.join(names_concat)

    # Check by category mapping
    for allergy in matched_allergies:
        restricted_categories = {c.lower() for c in ALLERGY_CATEGORY_MAP.get(allergy, set())}
        if categories & restricted_categories:
            return True

    # Check by name keywords
    for allergy in matched_allergies:
        for keyword in ALLERGY_NAME_KEYWORDS.get(allergy, []):
            if keyword in names_text:
                return True

    return False

def filter_recipes_by_preferences(recipes, dietary_needs):
    """Filter recipes based on dietary needs using module-level DIET_MAP"""
    if not dietary_needs:
        return recipes

    dietary_needs_lower = dietary_needs.lower()
    restrictions = set()
    
    # Collect all restricted ingredients for the given dietary needs
    for diet_key, restricted_items in DIET_MAP.items():
        if diet_key in dietary_needs_lower:
            restrictions.update(restricted_items)

    filtered_recipes = []
    for recipe in recipes:
        ingredients = recipe['ingredients'] or []
        normalized_ingredients = normalize_ingredient_data(ingredients)
        
        # Check if any restricted ingredient is present
        violates = False
        for ing in normalized_ingredients:
            ing_name = ing['name'].lower()
            ing_category = ing['category'].lower()
            
            # Check if ingredient name contains any restricted term
            if any(restricted in ing_name for restricted in restrictions):
                violates = True
                break
            # Check if category contains any restricted term  
            if any(restricted in ing_category for restricted in restrictions):
                violates = True
                break
        
        if not violates:
            filtered_recipes.append(recipe)
    
    return filtered_recipes

# Neo4j Recipe Recommendation function
def get_recipe_recommendations(ingredient_names, dietary_needs="", allergies=""):
    with driver.session() as session:
        # Use user ingredients
        cleaned_ingredients = [ing.lower().strip() for ing in ingredient_names if ing.strip()]
        
        print(f"DEBUG: User provided: {ingredient_names}")
        print(f"DEBUG: Using directly: {cleaned_ingredients}")
        
        if not cleaned_ingredients:
            return []
        
        # Get shelf life data
        shelf_life_data = session.run("""
            UNWIND $ingredients AS ingredient
            MATCH (i:Ingredient)
            WHERE toLower(trim(i.name)) = toLower(trim(ingredient))
            RETURN i.name AS name, i.shelf_life AS shelf_life, i.category AS category
            """, ingredients=cleaned_ingredients)
        
        shelf_life_map = {}
        matched_ingredients = []
        for record in shelf_life_data:
            name = record["name"]
            shelf_life = record["shelf_life"]
            category = record["category"]
            days = parse_shelf_life(shelf_life)
            shelf_life_map[name.lower()] = {
                "shelf_life": shelf_life,
                "days_remaining": days,
                "category": category
            }
            matched_ingredients.append(name.lower())
            # print(f"🍀 DEBUG: Exact shelf life match: '{name}' -> {shelf_life}")
        
        # Show which ingredients didn't match
        unmatched = [ing for ing in cleaned_ingredients if ing.lower() not in matched_ingredients]
        # if unmatched:
        #     print(f"🍄 DEBUG: No shelf life data for: {unmatched}")

        result = session.run("""
            // Find recipes that use EXACT matches of user's ingredients
            MATCH (r:Recipe)-[:USES]->(i:Ingredient)
            WHERE toLower(trim(i.name)) IN $ingredients
            
            // Count exact matches
            WITH r, 
                 collect(DISTINCT i.name) AS matchedIngredientNames,
                 size([(r)-[:USES]->(ing) | ing]) AS totalIngredients
            
            RETURN r.title AS title,
                   r.id AS recipe_id,
                   [(r)-[:USES]->(i) | {name: i.name, category: i.category, shelf_life: i.shelf_life}] AS ingredients,
                   size(matchedIngredientNames) AS matches,
                   totalIngredients,
                   matchedIngredientNames AS matchedNames,
                   coalesce(r.directions, []) AS steps
            ORDER BY size(matchedIngredientNames) DESC, totalIngredients ASC
            LIMIT 20
            """, 
            ingredients=[ing.lower() for ing in cleaned_ingredients],
        )
        
        all_recipes = []
        user_ingredient_count = len(cleaned_ingredients)
        
        for record in result:
            recipe = dict(record)
            
            # Calculate coverage percentages
            user_coverage = recipe["matches"] / float(user_ingredient_count) if user_ingredient_count > 0 else 0
            recipe_coverage = recipe["matches"] / float(recipe["totalIngredients"]) if recipe["totalIngredients"] > 0 else 0
            
            recipe["user_coverage"] = user_coverage
            recipe["recipe_coverage"] = recipe_coverage
            
            # Track exactly which ingredients are used
            used_ingredients = recipe["matchedNames"]
            urgent_ingredients = []
            total_urgency_score = 0
            
            for ing_name in used_ingredients:
                # Calculate urgency for matched ingredients
                ing_data = shelf_life_map.get(ing_name.lower())
                if ing_data:
                    days_remaining = ing_data["days_remaining"]
                    if days_remaining < 7:
                        urgent_ingredients.append(f"{ing_name} ({int(days_remaining)} days)")
                    if days_remaining < float('inf'):
                        total_urgency_score += 10.0 / (days_remaining + 1)
            
            recipe["used_ingredients"] = used_ingredients
            recipe["urgent_ingredients"] = urgent_ingredients
            recipe["urgency_score"] = total_urgency_score
            recipe["urgent_count"] = len(urgent_ingredients)
            
            # print(f"🦋🦋 DEBUG: '{recipe['title']}' - uses {recipe['matches']} ingredients: {used_ingredients}")
            all_recipes.append(recipe)
        
        # print(f"🌼 DEBUG: Found {len(all_recipes)} recipes before filtering")
        
        # Apply dietary filtering
        filtered_recipes = filter_recipes_by_preferences(all_recipes, dietary_needs)
        # print(f"DEBUG: {len(filtered_recipes)} recipes after dietary filtering")

        # Apply allergy filtering
        if allergies:
            before_allergy = len(filtered_recipes)
            filtered_recipes = [r for r in filtered_recipes if not violates_allergies(r['ingredients'], allergies)]
            # print(f"DEBUG: {len(filtered_recipes)} recipes after allergy filtering (removed {before_allergy - len(filtered_recipes)})")

        # Prioritize recipes that use the most ingredients
        if filtered_recipes:
            filtered_recipes.sort(key=lambda x: (
                -x["user_coverage"],  # Use most of user's ingredients
                -x["matches"],        # Maximize number of ingredients used
                -x["urgency_score"],  # Consider expiration
                x["totalIngredients"] # Prefer simpler recipes
            ))

        # print(f"DEBUG: Returning {len(filtered_recipes)} final recipes")
        return filtered_recipes[:10]
    
# Initialize the LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0.7,
    streaming=True
)

# Initialize the prompt template
recipe_prompt = PromptTemplate(
    input_variables=["ingredients", "dietary_needs", "recipe_results", "allergies"],
    template="""
    You're a professional chef assistant helping reduce food waste.
    Recommend recipes that uses as many as possible ingredients as the user has 
    available in their pantry, as given by the database below.

    CRITICAL CONSTRAINTS: 
    - DO NOT invent or modify recipes.
    - DO NOT add ingredients that aren't in the user's pantry.
    - If a recipe uses ingredients the user doesn't have, acknowledge this limitation.


    Available ingredients in user's pantry:
    {ingredients}

    Dietary preferences: {dietary_needs}
    Allergies: {allergies}

    Top matching recipes from database:
    {recipe_results}

    Format your response with:
    For each recipe:
    1. **Recipe Name** (Match Percentage)
    - From your pantry: [list of matching ingredients]
    * Missing: [list any ingredients explicitly required by the recipe that the user does not have in their pantry]
    - Instructions: [from the database]

    Only recommend recipes that do not contain ingredients conflicting with any of the user's allergies given above
    Include a tip about prioritizing any remaining ingredients in the user's pantry by urgency.
"""
)

# format recipe results for the LLM prompt
def format_results(recipes):
    recipe_results = []
    for r in recipes:
        title = r['title']
        coverage = r.get('user_coverage', 0)
        ingredient_names = [ing['name'] if isinstance(ing, dict) else ing for ing in r['ingredients'] ]
        urgent = r.get('urgent_ingredients', [])

        parts = [f"**{title}** ({coverage:.0%} match)"]
        parts.append(f"- Uses: {', '.join(ingredient_names) if ingredient_names else 'None'}")
        if urgent:
            parts.append(f"- URGENT: {', '.join(urgent)}")

        steps = r['steps']
        if steps:
            parts.append("- Instructions: " + ' '.join(steps))
        recipe_results.append('\n'.join(parts))

    return "\n\n".join(recipe_results)


# generate a recommendation string for given pantry/context
def generate_recommendation(pantry, dietary_needs="", allergies=""):
    """
    This function is safe to call from tests and will not run the Streamlit UI.
    """
    # Normalize pantry to list of strings
    # pantry_list = pantry if isinstance(pantry, (list, tuple)) else [x.strip() for x in str(pantry).split(',') if x.strip()]
    pantry_list = pantry if isinstance(pantry, (list, tuple)) else [x.strip() for x in str(pantry).split(',') if x.strip()]

    # Get recipe matches from KG
    allergies = ""
    # Try to get allergies from session state if available
    try:
        allergies = getattr(st.session_state, "allergies", "")
    except Exception:
        allergies = ""  
 
    try:
        recipes = get_recipe_recommendations(pantry_list, dietary_needs or "", allergies or "")
    except Exception:
        return []

    recipe_results = format_results(recipes)

    prompt_text = recipe_prompt.format(
        ingredients=", ".join(pantry_list),
        dietary_needs=(dietary_needs or "none"),
        allergies=allergies or "",
        recipe_results=recipe_results
    )

    # Call the LLM and return content
    try:
        response = llm([HumanMessage(content=prompt_text)])
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        return e

# Streamlit UI code
def run_streamlit_app():
    st.set_page_config(layout="wide")
    st.title("Pantry to Plate - Recipe Recommender")
    st.info("""This chatbot interface is designed to help you find recipes 
    that would prioritize the ingredients you have in your pantry.""")

    with st.sidebar:
        st.sidebar.header("How this works")
        st.markdown("""
        1. Add ingredients you have
        2. Set your dietary preferences
        3. Get smart recommendations that prioritize:
           - Using ingredients that expire soon
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
        
        st.session_state.allergies = st.text_input("Allergies")

    # Chat interface
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask for recipe recommendations"):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        # Determine if this is a new recipe request or a follow-up about an existing recipe
        prompt_lower = prompt.lower()
        recipe_keywords = ["recipe", "cook", "make", "suggest", "recommend", "ideas"]
        # follow-up indicators (ask for more details about a shown recipe)
        followup_indicators = ["how", "elaborate", "directions", "instruction", "instructions", "step", "steps", "detail", "is it"]

        contains_recipe_keyword = any(k in prompt_lower for k in recipe_keywords)

        # Check if user references one of the recently shown recipe titles
        referenced_title = None
        if st.session_state.get('current_recipes'):
            for r in st.session_state.current_recipes:
                title = r.get('title', '')
                if title and title.lower() in prompt_lower:
                    referenced_title = r
                    break

        is_followup = referenced_title is not None or any(fi in prompt_lower for fi in followup_indicators)

        with st.chat_message("assistant"):
            # If it's a clear new recipe request and pantry is provided, fetch recommendations
            if contains_recipe_keyword and not is_followup and st.session_state.get('pantry'):
                recipes = get_recipe_recommendations(
                    st.session_state.pantry,
                    st.session_state.dietary_needs,
                    st.session_state.allergies or ""
                )
                st.session_state.current_recipes = recipes

                # Format results with urgency info
                recipe_results = []
                for r in recipes:
                    recipe_results.append(
                        f"**{r['title']}** ({r['user_coverage']:.0%} match)\n"
                        f"- Uses: {', '.join(i['name'] if isinstance(i, dict) and 'name' in i else str(i) for i in r['ingredients'])}\n"
                        f"{('- URGENT: ' + ', '.join(r['urgent_ingredients'])) if r['urgent_ingredients'] else ''}\n"
                    )

                # Generate LLM response with recipe results
                response = llm([
                    HumanMessage(content=recipe_prompt.format(
                        ingredients=", ".join(st.session_state.pantry),
                        dietary_needs=st.session_state.dietary_needs or "none",
                        allergies=st.session_state.allergies or "",
                        recipe_results="\n\n".join(recipe_results)
                    ))
                ])

                full_response = response.content if hasattr(response, 'content') else str(response)

            else:
                # Treat as a follow-up or general chat: include recipe context if available
                enhanced_prompt = prompt
                if referenced_title:
                    # Provide detailed context for the referenced recipe
                    r = referenced_title
                    uses = ', '.join(i['name'] if isinstance(i, dict) and 'name' in i else str(i) for i in r['ingredients'])
                    steps = '\n'.join(r['steps']) if r['steps'] else ''
                    urgent = ', '.join(r['urgent_ingredients']) if r['urgent_ingredients'] else ''
                    recipe_detail = f"Recipe detail for {r['title']}: Uses: {uses}. {('- URGENT: ' + urgent) if urgent else ''} Directions: {steps}"
                    enhanced_prompt = f"{prompt}\n\nContext: {recipe_detail}"
                elif st.session_state.get('current_recipes'):
                    # Provide a compact context of recently shown recipes
                    recipe_titles = [r['title'] for r in st.session_state.current_recipes]
                    recipe_context = f"Recent recipes shown: {', '.join(recipe_titles)}"
                    enhanced_prompt = f"{prompt}\n\nContext: {recipe_context}"

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


if __name__ == "__main__":
    run_streamlit_app()