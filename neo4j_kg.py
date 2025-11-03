import os
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
import ast
from fuzzywuzzy import fuzz

# Load environment variables
load_dotenv()

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Load RecipeNLG dataset
recipes_df = pd.read_csv('./data/recipe_nlp_1000.csv')
recipes_df['id'] = recipes_df['id'].astype(str)

# Load Shelf Life dataset
shelf_life_df = pd.read_csv('./data_processed/foodkeeper_shelf_life_processed.csv')
# Normalize ingredient strings safely
shelf_life_df['Ingredient'] = shelf_life_df['Ingredient'].astype(str).str.lower().str.strip()


def find_shelf_life(ingredient):
    """Finds the best shelf life match using exact + fuzzy matching"""
    ingredient = ingredient.lower().strip()
    
    # Proper empty check
    match = shelf_life_df[shelf_life_df['Ingredient'] == ingredient]
    if not match.empty:
        return match.iloc[0]

    # Work on a copy to avoid modifying original DataFrame
    shelf_life_copy = shelf_life_df.copy()
    shelf_life_copy['similarity'] = shelf_life_copy['Ingredient'].apply(
        lambda x: fuzz.ratio(ingredient, str(x))
    )
    
    best_match = shelf_life_copy.nlargest(1, 'similarity')
    
    if not best_match.empty and best_match.iloc[0]['similarity'] >= 75:
        return best_match.iloc[0]
    
    return None

# Function to create Knowledge graph 
def create_knowledge_graph(tx, recipe):
    # Create Recipe node
    tx.run("""
    MERGE (r:Recipe {id: $id})
    SET r.title = $title
    """, 
    id=recipe['id'],
    title=recipe.get('title', ''),
   )
    
    '''
    Ingredients can be a list or string representation of list
    Parses recipe['ingredients']
    If ingredients is a string (likely a stringified list like "['egg','milk']"),
    uses ast.literal_eval to turn it into a Python list; otherwise expects it to already be iterable.
    '''
    ingredients = ast.literal_eval(recipe['ingredients']) if isinstance(recipe['ingredients'], str) else recipe['ingredients']
    
    '''
    Calls find_shelf_life(ing) to get matching shelf-life data.
    '''
    for ing in ingredients:
        ing = ing.strip()
        if not ing:
            continue
            
        
        shelf_data = find_shelf_life(ing)
        
        # Create Ingredient node with shelf life
        tx.run("""
        MATCH (r:Recipe {id: $recipe_id})
        MERGE (i:Ingredient {name: $name})
        SET i.category = $category,
            i.shelf_life = $shelf_life,
            i.source = $source
        MERGE (r)-[:USES]->(i)
        """,
        recipe_id=recipe['id'],
        name=ing,
        category=shelf_data['Category'] if shelf_data else '',
        shelf_life=shelf_data['Shelf_Life'] if shelf_data else '',
        source='FoodKeeper' if shelf_data else 'RecipeNLG')
    
    # Process Steps
    directions = ast.literal_eval(recipe['directions']) if isinstance(recipe['directions'], str) else recipe['directions']
    for step_num, step in enumerate(directions, 1):
        tx.run("""
        MATCH (r:Recipe {id: $recipe_id})
        MERGE (s:Step {description: $description})
        SET s.order = $order
        MERGE (r)-[:HAS_STEP {order: $order}]->(s)
        """,
        recipe_id=recipe['id'],
        description=step.strip(),
        order=step_num)

def main():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    except Exception as e:
        print(f"Failed to create Neo4j driver: {e}")
        return


    with driver.session() as session:
        for _, recipe in recipes_df.iterrows():
            try:
                session.execute_write(create_knowledge_graph, recipe)
            except Exception as e:
                print(f"Error with recipe {recipe['id']}: {str(e)}")

    try:
        if driver is not None:
            driver.close()
    except Exception:
        pass
    print("Knowledge graph created")


if __name__ == "__main__":
    main()