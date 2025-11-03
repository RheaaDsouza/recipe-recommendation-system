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
        # get all shelf life data for the ingredients
        ingredients_for_query = [str(x).lower() for x in ingredient_names]
        shelf_life_data = session.run("""
            UNWIND $ingredients AS ingredient
            MATCH (i:Ingredient)
            WHERE toLower(trim(i.name)) = toLower(trim(ingredient))
            RETURN i.name AS name, i.shelf_life AS shelf_life
            """, ingredients=ingredients_for_query)
        
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
            
            // Finding recipes using the ingredients
            MATCH (r:Recipe)-[:USES]->(i:Ingredient)
            WHERE toLower(trim(i.name)) IN searchIngredients
            
            // Grouping by recipe and count matches
            WITH r, searchIngredients, collect(i) AS usedIngredients, 
                 size([(r)-[:USES]->(x) | x]) AS totalIngredients

            
            RETURN r.title AS title,
             [(r)-[:USES]->(i) | {name: i.name, category: i.category}] AS ingredients,
                   size(usedIngredients) AS matches,
                   totalIngredients,
                   coalesce(r.directions, []) AS steps
            ORDER BY size(usedIngredients) DESC, totalIngredients ASC
            LIMIT 20
            """, 
            ingredients=ingredients_for_query,
        )
        
        all_recipes = []
        for record in result:
            recipe = dict(record)
            recipe["coverage"] = recipe["matches"] / float(recipe["totalIngredients"])
            
            # Calculate urgency information (use category-aware ingredient representations)
            urgent_ingredients = []
            urgency_score = 0
            # recipe['ingredients'] may be a list of dicts or list of strings (look into this)
            for ing in recipe["ingredients"]:
                if isinstance(ing, dict):
                    ing_name = ing['name']
                else:
                    ing_name = str(ing)

                ing_key = ing_name.lower()
                ing_data = shelf_life_map.get(ing_key)
                if ing_data and ing_data["days_remaining"] < 7:
                    urgent_ingredients.append(f"{ing_name} ({ing_data['days_remaining']} days)")
                if ing_data:
                    urgency_score += 1.0 / (ing_data["days_remaining"] + 1)
            
            recipe["urgent_ingredients"] = urgent_ingredients
            recipe["urgency_score"] = urgency_score
            all_recipes.append(recipe)
        
        # Apply dietary filtering ONCE
        filtered_recipes = filter_recipes_by_preferences(all_recipes, dietary_needs)

        # Apply allergy filtering
        if allergies:
            filtered_recipes = [r for r in filtered_recipes if not violates_allergies(r['ingredients'], allergies)]

        # Sort by urgency score then coverage
        filtered_recipes.sort(key=lambda x: (-x["urgency_score"], -x["coverage"]))
        return filtered_recipes[:20]

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
    Recommend recipes prioritizing ingredients that will expire soon.
    
    Available ingredients:
    {ingredients}
    
    Dietary preferences: {dietary_needs}
    Allergies: {allergies}

    Top matching recipes from database (indicates soon-to-expire ingredients):
    {recipe_results}
    
    Format your response with:
    For each recipe:
    1. **Recipe Name** (Match Percentage)
       - Uses: [list of matching ingredients]
       - Instructions / Directions of the recipe
    
    Also include a final tip on the ingredients that need to be prioritized due 
    to their short shelf life.
    """
)

# format recipe results for the LLM prompt
def format_results(recipes):
    recipe_results = []
    for r in recipes:
        title = r['title']
        coverage = r.get('coverage', 0)
        ingredient_names = [ing['name'] if isinstance(ing, dict) else ing for ing in r['ingredients'] ]
        urgent = r['urgent_ingredients']

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
        import streamlit as st
        allergies = getattr(st.session_state, "allergies", "")
    except Exception:
        allergies = ""
    try:
        recipes = get_recipe_recommendations(pantry_list, dietary_needs or "", allergies or "")
    except Exception:
        recipes = []

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
                title = r['title']
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
                        f"**{r['title']}** ({r['coverage']:.0%} match)\n"
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